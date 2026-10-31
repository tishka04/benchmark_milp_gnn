"""Plot the already computed statistical audit (never runs experiments)."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.analyze_criticality_existing import FAMILIES, LABELS

COLORS=["#2c78ad", "#d78d2c", "#c54b4c", "#169685", "#835ca4", "#5a6877"]
SHORT={"unserved_log":"Énergie non servie", "runtime_log":"Temps MILP (plafond 600 s)",
       "ref_timeout":"Limite de temps MILP", "jepa_fail120":"Échec JEPA à 120 s",
       "pipeline_fail120":"Échec pipeline à 120 s", "pipeline_bad10":"Pipeline : qualité > +10 % ou échec"}
MOD_LABEL={"size":"Taille seule", "size_crit":"+ Criticité", "size_stress":"+ Stress", "size_hard":"+ Hardness",
           "size_both":"+ Stress et hardness", "size_components":"+ 18 composantes hors taille"}
PRIMARY=["unserved_log","runtime_log","ref_timeout","jepa_fail120"]
FEATURE_LABEL={
"A1_vre_penetration":"A1  Pénétration ENR", "A2_residual_volatility":"A2  Volatilité résiduelle",
"A3_peak_to_valley":"A3  Pic / creux", "A4_short_term_variability":"A4  Variabilité météo",
"A5_demand_scale":"A5  Niveau de demande", "A6_peak_to_firm":"A6  Pic / capacité ferme",
"A7_inv_storage_power":"A7  Manque de puissance stockage", "A8_inv_storage_energy":"A8  Manque d'énergie stockage",
"A9_inv_thermal_flex":"A9  Manque de flexibilité thermique", "A10_inv_dr_headroom":"A10  Manque de marge DR",
"A11_trade_reliance":"A11  Dépendance aux échanges", "A12_inv_congestion":"A12  Proxy de congestion",
"B4_thermal_density":"B4  Densité thermique", "B6_startup_intensity":"B6  Coût relatif de démarrage",
"B7_ramp_tightness":"B7  Rigidité des rampes", "B8_soc_tightness":"B8  Rigidité du SOC",
"B9_interconn_density":"B9  Densité des interconnexions", "B10_network_hetero":"B10  Hétérogénéité du réseau"}


def title(fig,main,sub):
    fig.text(.065,.965,main,fontsize=19,fontweight="bold",va="top")
    fig.text(.065,.915,sub,fontsize=10.5,va="top",color="#566476")


def save(fig,folder,name):
    fig.savefig(folder/(name+".png"),dpi=190)
    fig.savefig(folder/(name+".svg"))
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--analysis",type=Path,default=Path("outputs/load_shifting_v1/evaluation/budgets_v1/criticality_audit_20261027"))
    args=ap.parse_args();p=args.analysis;out=p/"figures";out.mkdir(exist_ok=True)
    d=pd.read_csv(p/"scenario_data.csv");a=pd.read_csv(p/"associations.csv");v=pd.read_csv(p/"validation.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":"#23354a",
        "axes.labelcolor":"#344458","axes.spines.top":False,"axes.spines.right":False,
        "axes.edgecolor":"#b9c4ce","grid.color":"#dce3e9","svg.fonttype":"none"})

    fig,axs=plt.subplots(1,2,figsize=(13,7.3))
    title(fig,"Deux axes distincts, des familles très structurées",
          "600 scénarios ; mêmes poids pour tous (alpha = 0,5). Les scores personnalisés des personas sont conservés séparément.")
    for f,c in zip(FAMILIES,COLORS):
        s=d[d.family==f]
        axs[0].scatter(s.stress,s.hardness,c=c,s=25,alpha=.65,label=LABELS[f],edgecolors="none")
        axs[1].scatter(s.n_zones,s.criticality,c=c,s=25,alpha=.65,edgecolors="none")
    axs[0].set(xlabel="Stress",ylabel="Hardness")
    axs[1].set(xlabel="Nombre de zones (échelle logarithmique)",ylabel="Criticité = (stress + hardness) / 2",xscale="log")
    for ax in axs:ax.grid(alpha=.5);ax.set_axisbelow(True)
    fig.legend(*axs[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.045))
    fig.subplots_adjust(left=.08,right=.97,top=.82,bottom=.23,wspace=.27)
    save(fig,out,"01_indices_et_taille")

    endpoints=["unserved_log","runtime_log","ref_timeout","pipeline_fail120","jepa_fail120"]
    collabels=["Énergie\nnon servie","Temps\nMILP","Limite\nMILP","Échec\npipeline","Échec\nJEPA"]
    fig,axs=plt.subplots(1,3,figsize=(13,5.9))
    title(fig,"Les corrélations s'atténuent après contrôle de la taille",
          "Corrélations de rang ; ajustement non linéaire de la taille, puis ajout des effets fixes de famille. Une ligne par scénario.")
    for ax,adj,label in zip(axs,["raw","size","size_family"],["Brutes","Après contrôle de la taille","Après taille + famille"]):
        z=a[(a.scope=="all600")&(a.adjustment==adj)]
        vals=np.array([[z[(z.feature==f)&(z.endpoint==e)].rho.iloc[0] for e in endpoints] for f in ["criticality","stress","hardness"]])
        im=ax.imshow(vals,cmap="RdBu_r",vmin=-.8,vmax=.8,aspect="auto")
        for i in range(3):
            for j in range(len(endpoints)):
                ax.text(j,i,f"{vals[i,j]:+.2f}",ha="center",va="center",color="white" if abs(vals[i,j])>.47 else "#23354a",fontsize=10)
        ax.set_xticks(range(5),collabels,fontsize=9)
        ax.set_yticks(range(3),["Criticité","Stress","Hardness"] if adj=="raw" else [""]*3)
        ax.set_title(label,fontsize=11,pad=13)
        ax.tick_params(length=0)
    fig.subplots_adjust(left=.085,right=.96,top=.72,bottom=.30,wspace=.18)
    cb=fig.colorbar(im,ax=axs,orientation="horizontal",fraction=.07,pad=.25,shrink=.5)
    cb.set_label("Corrélation : négative ← 0 → positive",fontsize=10)
    save(fig,out,"02_correlations_ajustees")

    fig,axs=plt.subplots(2,2,figsize=(13,9.2))
    title(fig,"L'information supplémentaire est surtout dans les composantes",
          "Gain de R² hors échantillon par rapport à la taille seule ; 5 plis × 10 répétitions. Intervalles bootstrap appariés à 95 %.")
    models=["size_crit","size_stress","size_hard","size_both","size_components"]
    for ax,e in zip(axs.flat,PRIMARY):
        z=v[(v.scheme=="random_family_balanced")&(v.endpoint==e)].set_index("model")
        for i,m in enumerate(models):
            r=z.loc[m];color="#167d80" if m=="size_components" else "#687f98"
            ax.errorbar(r.delta_r2,i,xerr=[[max(0,r.delta_r2-r.delta_low)],[max(0,r.delta_high-r.delta_r2)]],
                        fmt="o",color=color,capsize=3,markersize=6)
        ax.axvline(0,color="#b7c2cc",lw=1)
        ax.set_yticks(range(5),[MOD_LABEL[m] for m in models],fontsize=9)
        ax.invert_yaxis();ax.set_xlim(-.035,.215);ax.set_ylim(4.6,-.6)
        ax.grid(axis="x",alpha=.65);ax.set_xlabel("Gain de R² (0 = aucun apport)")
        ax.set_title(f"{SHORT[e]}\nR² taille seule = {z.loc['size','r2_cv']:.3f}",fontsize=11,pad=9)
    fig.subplots_adjust(left=.22,right=.97,top=.81,bottom=.11,hspace=.55,wspace=.72)
    fig.text(.065,.025,"Régressions exploratoires sur les résultats sauvegardés. Les 18 composantes sont ajustées conjointement : ce n'est pas le score du manuscrit.",fontsize=10)
    save(fig,out,"03_apport_hors_echantillon")

    fig,ax=plt.subplots(figsize=(13,9.4))
    title(fig,"Les rampes dominent, avec des signes opposés selon le résultat",
          "Corrélations de rang après contrôle de la taille et de la famille. * : q < 0,05 après correction de Benjamini-Hochberg.")
    ends=["unserved_log","runtime_log","ref_timeout","pipeline_fail120","jepa_fail120","pipeline_bad10"]
    features=list(FEATURE_LABEL)
    z=a[(a.scope=="all600")&(a.adjustment=="size_family")]
    vals=np.array([[z[(z.feature==f)&(z.endpoint==e)].rho.iloc[0] for e in ends] for f in features])
    im=ax.imshow(vals,cmap="RdBu_r",vmin=-.65,vmax=.65,aspect="auto")
    for i,f in enumerate(features):
        for j,e in enumerate(ends):
            q=z[(z.feature==f)&(z.endpoint==e)].q_bh.iloc[0]
            ax.text(j,i,f"{vals[i,j]:+.2f}"+("*" if q<.05 else ""),ha="center",va="center",fontsize=9,
                    color="white" if abs(vals[i,j])>.39 else "#23354a")
    ax.set_yticks(range(len(features)),[FEATURE_LABEL[f] for f in features],fontsize=10)
    ax.set_xticks(range(len(ends)),["Énergie non\nservie","Temps\nMILP","Limite\nMILP","Échec\npipeline","Échec\nJEPA","Qualité pipeline\n> +10 % / échec"],fontsize=10)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=.32,right=.92,top=.83,bottom=.15)
    cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.025);cb.set_label("Corrélation ajustée")
    fig.text(.065,.035,"B1/B2/B3 : taille déjà contrôlée ; B5 : constant. Associations exploratoires, non causales ; l'effet de formulation des rampes est documenté séparément.",fontsize=9.5)
    save(fig,out,"04_composantes")

    fig,axs=plt.subplots(2,2,figsize=(13,8.8))
    title(fig,"Le transfert à une famille absente reste fragile",
          "Même régression et mêmes variables ; comparaison entre plis mélangés et exclusion complète d'une famille de l'ajustement.")
    ms=["size","size_crit","size_both","size_components"]
    for ax,e in zip(axs.flat,PRIMARY):
        x=np.arange(4)
        for delta,scheme,col,label in [(-.18,"random_family_balanced","#7b92ac","Familles présentes"),(.18,"leave_one_family_out","#d07b38","Famille exclue")]:
            vals=[v[(v.scheme==scheme)&(v.endpoint==e)&(v.model==m)].r2_cv.iloc[0] for m in ms]
            bars=ax.bar(x+delta,vals,width=.34,color=col,label=label)
            for rect,value in zip(bars,vals):
                ax.annotate(f"{value:.2f}",(rect.get_x()+rect.get_width()/2,value),xytext=(0,4 if value>=0 else -5),textcoords="offset points",ha="center",va="bottom" if value>=0 else "top",fontsize=8.5)
        ax.set_xticks(x,["Taille","+ Criticité","+ Stress\net hardness","+ 18\ncomposantes"],fontsize=9)
        ax.set_title(SHORT[e],fontsize=11,pad=10)
        ax.axhline(0,color="#a5b3c2",lw=.8);ax.set_ylabel("R² hors échantillon");ax.grid(axis="y",alpha=.5);ax.set_axisbelow(True)
        ymin=min(0,min(b.get_height() for b in ax.patches));ymax=max(.6,max(b.get_height() for b in ax.patches))
        ax.set_ylim(ymin-.16*(ymax-ymin),ymax+.2*(ymax-ymin))
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.055))
    fig.subplots_adjust(left=.085,right=.97,top=.82,bottom=.19,hspace=.60,wspace=.28)
    fig.text(.065,.025,"Un R² négatif indique une erreur supérieure à celle de la moyenne globale. Scénarios synthétiques ; aucun transfert à un réseau réel n'est testé.",fontsize=10)
    save(fig,out,"05_transfert")

    fig,axs=plt.subplots(1,2,figsize=(13,7.2))
    title(fig,"Une rampe trop faible interdit le démarrage thermique dans ce modèle",
          "Production initiale nulle + minimum thermique 45 % + rampe sans exception de démarrage. Lecture des dispatchs déjà calculés.")
    for f,c in zip(FAMILIES,COLORS):
        z=d[d.family==f]
        axs[0].scatter(z.thermal_ramp_pct,z.unserved_pct,c=c,s=20,alpha=.6,label=LABELS[f],edgecolors="none")
        axs[1].scatter(z.thermal_ramp_pct,np.minimum(z.ref_seconds,600),c=c,s=20,alpha=.6,edgecolors="none")
    axs[0].set(xlabel="Rampe thermique / capacité",ylabel="Énergie non servie / demande (%)",yscale="symlog")
    axs[0].set_yscale("symlog",linthresh=.01)
    axs[1].set(xlabel="Rampe thermique / capacité",ylabel="Temps MILP de référence (s)",yscale="log")
    for ax in axs:
        ax.axvline(.45,color="#ae3f40",ls="--",lw=1.5);ax.grid(alpha=.45);ax.set_axisbelow(True)
    fig.legend(*axs[0].get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.055))
    fig.subplots_adjust(left=.085,right=.97,top=.81,bottom=.24,wspace=.28)
    fig.text(.065,.025,"255 scénarios sous le seuil : tous sans production thermique, tous résolus à l'optimum ; 146 des 148 cas avec énergie non servie sont dans ce groupe.",fontsize=10)
    save(fig,out,"06_effet_formulation")
    print(out.resolve())


if __name__=="__main__":main()
