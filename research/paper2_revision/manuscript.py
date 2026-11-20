"""Publish verified analysis artifacts and build a portable analysis/manuscript bundle."""
from pathlib import Path
import shutil
import subprocess
import zipfile
import json
import re
import pandas as pd
from .core import ROOT, HOME, read, verify, sha, config

MANUSCRIPTS=ROOT/"research/three_papers/manuscripts"

def latest_analysis(run):
    receipts=list((Path(run)/"analyses").glob("*/receipt.json"))
    if not receipts: raise ValueError("Run AnalyzeExisting or Analyze first")
    receipts.sort(key=lambda p:(read(p).get("name")=="complete",p.stat().st_mtime_ns),reverse=True)
    p=receipts[0]; receipt=verify(read(p))
    for name,h in receipt["outputs"].items():
        if sha(p.parent/name)!=h: raise ValueError("Analysis export changed")
    return p.parent,receipt

def update(run, *, compile_pdf=True):
    folder,receipt=latest_analysis(run)
    dest=MANUSCRIPTS/"paper2/results"; dest.mkdir(parents=True,exist_ok=True)
    for p in (folder/"figures").glob("*.*"): shutil.copy2(p,dest/p.name)
    df=pd.read_csv(folder/"tables/coverage.csv")
    historical=df[(df.partition=="historical_development")&(df.repetition==0)&(df.family=="all")&(df.budget==120)]
    names={"highs":"HiGHS","rh":"RH","jepa":"JEPA","pipeline":"Pipeline"}
    out=[r"\begin{table}[htbp]\centering\small",r"\begin{tabular}{lrrrr}\toprule",
         r"Method & Valid & No ENS & Cost $\leq5\%$ & Joint\\\midrule"]
    for method in names:
        row=historical[historical.method==method].set_index("endpoint")
        out.append(names[method]+" & "+" & ".join(str(int(row.loc[e,"successes"])) for e in ("valid","continuity","economic_5","joint_5"))+r"\\")
    out.extend([r"\bottomrule\end{tabular}",r"\caption{Primary historical outcomes at 120 s; counts out of 600. Every count requires a valid dispatch and complete exit before the deadline. The joint criterion combines economic quality and ENS $\leq10^{-3}$ MWh.}\label{tab:coverage}",r"\end{table}"])
    intervals=[]
    for method in names:
        q=historical[(historical.method==method)&(historical.endpoint=="joint_5")].iloc[0]
        intervals.append(f"{names[method]} {100*q.estimate:.1f}\\% [{100*q.lower:.1f}, {100*q.upper:.1f}]")
    out.append("Joint coverage and pointwise 95\\% scenario-bootstrap intervals are "+"; ".join(intervals)+". These intervals describe resampling within the six constructed families.")
    out.append("A degenerate interval at 100\\% observed coverage is not a zero-risk deployment guarantee.")
    (dest/"historical_table.tex").write_text("\n".join(out)+"\n",encoding="utf-8")
    extension=df[(df.partition=="final")&(df.repetition==0)&(df.family=="all")&(df.endpoint=="joint_5")]
    observed=extension[extension.n>0]
    if observed.empty:
        future=("The independent comparison, matched energy ablation, structural transformations, variable-horizon "
                "training and external-solver comparison have not yet been executed in this revision. "
                "Their protocol is specified in Section~\\ref{sec:extensions}; no performance conclusion is drawn from these designs.\n")
    else:
        lines=[r"\begin{table}[htbp]\centering\small",r"\begin{tabular}{llrrr}\toprule",
               r"Suite / method & Deadline & Successes & Known & Unknown\\\midrule"]
        for _,r in extension.sort_values(["suite","method","budget"]).iterrows():
            name=r["method"].replace("_",r"\_")
            lines.append(f"{r['suite']} / {name} & {int(r.budget)} s & {int(r.successes)} & {int(r.n)} & {int(r.unknown)}"+r"\\")
        lines += [r"\bottomrule\end{tabular}",r"\caption{Independent-test joint outcomes. Missing observations remain unknown; paired energy-ablation runs are a separate suite.}\label{tab:independent}",r"\end{table}"]
        primary=pd.read_csv(folder/"tables/primary_inference.csv")
        lines.append(f"Of the 25 prespecified comparisons, {int(primary.complete.sum())} have complete paired support; "
                     f"{int(primary.significant.sum())} reject the equal-coverage null after Holm adjustment. "
                     "Incomplete contrasts do not establish a difference. Effect direction and magnitude must be read together with their intervals.")
        # Include paired effects and uncertainty, not p-values alone.
        lines.extend([r"\begin{table}[htbp]\centering\scriptsize",r"\begin{tabular}{llrr}\toprule",
            r"Contrast & Deadline & Coverage difference [95\% CI] & Holm $p$\\\midrule"])
        for _,r in primary[primary.complete].iterrows():
            a=r.method_a.replace("_",r"\_"); b=r.method_b.replace("_",r"\_")
            lines.append(f"{a} -- {b} & {int(r.budget)} s & {100*r.estimate:.1f} [{100*r.lower:.1f}, {100*r.upper:.1f}] & {r.p_holm:.3g}"+r"\\")
        lines.extend([r"\bottomrule\end{tabular}",r"\caption{Prespecified independent-test contrasts, in percentage points. Intervals are pointwise; the 25-test family is controlled by Holm-adjusted exact paired tests.}\end{table}"])
        future="\n".join(lines)+"\n"
    (dest/"extension_results.tex").write_text(future,encoding="utf-8")
    if observed.empty:
        status=r"\newcommand{\pIIExtensionStatus}{Those extensions remain unexecuted in this revision.}"+"\n"
        status+=r"\newcommand{\pIIIntroStatus}{The latter two are specified for independent evaluation and remain pending.}"+"\n"
    else:
        status=r"\newcommand{\pIIExtensionStatus}{Independent outcomes available so far are reported separately with explicit support.}"+"\n"
        status+=r"\newcommand{\pIIIntroStatus}{The extensions are assessed separately, with missing observations kept explicit.}"+"\n"
    (dest/"evidence_status.tex").write_text(status,encoding="utf-8")
    provenance=dict(analysis=str(folder),receipt_sha256=sha(folder/"receipt.json"),source=receipt["source"],
                    publication="No new solver or training result is inferred from an absent record.")
    (dest/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    validate_tex()
    build=dict(compiled=False,reason="Compilation not requested")
    if compile_pdf: build=compile_manuscript()
    bundle=analysis_bundle(run,folder)
    return dict(results=str(dest),build=build,bundle=str(bundle))

def validate_tex():
    main=MANUSCRIPTS/"paper2_methods_generalization.tex"
    files=[main,*list((MANUSCRIPTS/"paper2").rglob("*.tex"))]
    content="\n".join(p.read_text(encoding="utf-8") for p in files)
    for forbidden in (r"\path{",r"\pending{",r"\tablepath{","C:\\Users", "src/", "runs/", "thermal_transitions_v1"):
        if forbidden in content: raise ValueError(f"Implementation detail leaked into manuscript: {forbidden}")
    if "Subsequent work" not in content: raise ValueError("Missing bridge to paper 3")
    bibliography=(MANUSCRIPTS/"paper2/references.bib").read_text(encoding="utf-8")
    keys=set(re.findall(r"@\w+\{([^,]+),",bibliography))
    citations={k.strip() for match in re.findall(r"\\cite\w*\{([^}]+)\}",content) for k in match.split(",")}
    if citations-keys: raise ValueError(f"Missing citations: {citations-keys}")

def compile_manuscript():
    output=ROOT/"output/pdf/paper2"; output.mkdir(parents=True,exist_ok=True)
    tectonic=shutil.which("tectonic")
    if not tectonic:
        candidate=ROOT/"tmp/paper1_revision_tools/tectonic.exe"
        if candidate.exists(): tectonic=str(candidate)
    latexmk=shutil.which("latexmk")
    if tectonic:
        command=[tectonic,"--keep-logs","--keep-intermediates","--outdir",str(output),"paper2_methods_generalization.tex"]
    elif latexmk:
        command=[latexmk,"-pdf","-interaction=nonstopmode",f"-outdir={output}","paper2_methods_generalization.tex"]
    else:
        return dict(compiled=False,reason="Install Tectonic or latexmk; LaTeX sources and figures are complete")
    try:
        proc=subprocess.run(command,cwd=MANUSCRIPTS,capture_output=True,text=True,timeout=180)
    except subprocess.TimeoutExpired:
        return dict(compiled=False,reason="Compilation exceeded 180 s")
    log=output/"build.txt"; log.write_text(proc.stdout+proc.stderr,encoding="utf-8")
    if proc.returncode: raise RuntimeError(f"LaTeX compilation failed; inspect {log}")
    return dict(compiled=True,pdf=str(output/"paper2_methods_generalization.pdf"),log=str(log))

def analysis_bundle(run,folder):
    """Portable reproduction of tables/figures and LaTeX; no absolute input paths needed."""
    run=Path(run); target=run/"paper2_analysis_reproduction.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for p in HOME.glob("*.py"): archive.write(p,"research/paper2_revision/"+p.name)
        for name in ("config.json","README.md","GUIDE_COMMANDES.md","run_paper2.ps1","verification.json","bibliography_verification.json","requirements-analysis.txt"):
            p=HOME/name
            if p.exists(): archive.write(p,"research/paper2_revision/"+name)
        for p in (HOME/"tests").glob("*.py"):
            archive.write(p,"research/paper2_revision/tests/"+p.name)
        archive.write(folder/"tables/observations.csv","observations.csv")
        archive.write(folder/"receipt.json","source_receipt.json")
        archive.write(MANUSCRIPTS/"preamble.tex","manuscripts/preamble.tex")
        archive.write(MANUSCRIPTS/"paper2_methods_generalization.tex","manuscripts/paper2_methods_generalization.tex")
        pdf=ROOT/"output/pdf/paper2/paper2_methods_generalization.pdf"
        if pdf.exists(): archive.write(pdf,"paper2_methods_generalization.pdf")
        for p in (MANUSCRIPTS/"paper2").rglob("*"):
            if p.is_file(): archive.write(p,"manuscripts/paper2/"+p.relative_to(MANUSCRIPTS/"paper2").as_posix())
        archive.writestr("REPRODUCE.txt","Extract the archive and use Python 3.11 in a separate environment:\npython -m pip install -r research/paper2_revision/requirements-analysis.txt\npython -B -m research.paper2_revision.reproduce\n\nThis bundle reproduces analysis and the manuscript. Running new optimization/training or physics tests additionally requires the original runtime, scenario sources, and frozen learned models. No solver binaries or licensed software are redistributed.\n")
    return target
