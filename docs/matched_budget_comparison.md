# Comparaison a budgets de temps egaux

Preparation independante de la campagne historique : aucun fichier source utilise
par ses workers n'est modifie. Le nouveau protocole est
`matched_budgets_gpu16_feasible_energy_v1`. Les resultats historiques conservent
leur selecteur d'origine et ne sont pas reutilises pour cette comparaison.

## Protocole retenu

- Budgets : **1, 2, 5, 10, 60, 120 secondes**, chacun independamment pour chaque
  scenario et methode, dans un nouveau processus. Une courbe n'est pas deduite
  des prefixes d'une seule execution de 120 secondes.
- **MILP** : scenario original, sans warm start, HiGHS autorise a utiliser
  **16 threads**. Les ameliorations de l'incumbent sont publiees pendant le solve,
  apres validation de la matrice originale, des bornes et de l'integralite.
  La derniere solution valide recue par le parent avant l'echeance est conservee.
- **Pipeline** : **16 candidats generes en un batch CUDA**, puis **16 processus
  de reparation CPU simultanes, a un thread HiGHS chacun**. La generation et les
  reparations partagent la meme echeance absolue : aucun budget n'est divise par K
  ni remis a zero entre les phases. BLAS et PyTorch sont limites a un thread
  dans chaque worker.
- **JEPA** : decodage direct de l'ancre sur GPU, un candidat, puis reparation
  autorisee a utiliser **16 threads HiGHS**. Pas d'echantillonnage EBM.
- Le plafond CPU est comparable ; il ne garantit pas une occupation effective
  identique. Le GPU des methodes apprises est une ressource supplementaire
  explicitement declaree. K=16 ne correspond a 16 processus CPU qu'apres le batch.
- Convention **a froid** : demarrage Python, imports, lecture des donnees,
  chargement des poids, calcul GPU, transfert CPU, construction, solve,
  validation/publication et selection sont dans le temps de decision. Les
  embeddings HTE precalcules et les entrainements restent hors budget.
  A 1 ou 2 secondes, l'initialisation peut consommer tout le budget.
- Aucun resultat lu/selectionne a l'echeance ou apres celle-ci n'est accepte.
  Arret des processus et ecriture du checkpoint sont comptes separement du
  temps de decision ; le temps total reel est egalement conserve. Les erreurs
  de programmation/I/O arretent la campagne, les timeouts sont des observations.
- L'ordre des trois methodes tourne selon le scenario et le budget. Les seeds
  sont deterministes par scenario ; les reparations du batch utilisent seed+i.
  Les caches systeme ne sont pas purges. Chaque methode termine avant la suivante.

## Correction de la selection

`scripts/feasible_candidate_selection.py` filtre d'abord les candidats dont
l'objectif de dispatch est fini : la serialisation des reparations rend deja
cet objectif infini lorsque le dispatch ou les controles DR sont invalides.
Le nouveau parent verifie aussi explicitement `success` et `dr_feasible`.
Il choisit ensuite l'energie EBM minimale parmi les candidats eligibles.
Si aucune energie eligible n'est finie, il conserve le premier candidat faisable.
Si aucun candidat n'est faisable, le resultat reste un echec.

Il s'agit toujours d'une selection par energie, pas d'une selection par cout reel
de dispatch. Le cout peut donc augmenter quand un nouveau candidat faisable a
une energie inferieure. Un candidat infaisable ne remplace jamais un candidat
faisable. Le module fournit aussi une adaptation locale au processus pour
l'evaluateur existant, sans modifier son fichier source.

## Lancement ulterieur

Depuis la racine du depot, apres la fin de la campagne actuelle :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_matched_budget_comparison.ps1 `
  -HighsThreads 16 -K 16
```

Reprise, avec les memes options si elles ont ete personnalisees :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_matched_budget_comparison.ps1 -Resume
```

La grille 1,2,5,10,60,120 est configuree par defaut. Pour une grille personnalisee,
appeler directement le script depuis PowerShell afin de transmettre un tableau :

```powershell
.\scripts\run_matched_budget_comparison.ps1 -Budgets @(1,2,5,10,60,120)
```

`-Preview` affiche le plan sans calcul ni creation de sorties. `-AnalyzeOnly`
reconstruit uniquement les tableaux a partir des checkpoints disponibles.
`-LimitPerFamily N` permet une future verification sur un petit sous-ensemble,
avec un dossier `-OutputDir` distinct de celui de la campagne complete.

Le parent prend le meme verrou `_pipeline/run.lock` que la campagne originale.
Il refuse de lancer les workers tant qu'elle tourne ; il ne l'arrete pas.
Le nouveau dossier par defaut est `outputs/load_shifting_v1/evaluation/budgets_v1`.

La reprise verifie les hashes des inputs, modeles, sources et des checkpoints,
puis conserve chaque experience terminee (scenario, budget, methode), y compris
les echecs de faisabilite et les timeouts. Une experience interrompue sans
checkpoint est seule recalculee. Les tentatives anterieures restent disponibles.
Les snapshots d'une tentative interrompue ne sont pas recuperes apres coup.

La reprise accepte aussi les revisions precisement identifiees qui corrigent
uniquement les sauvegardes et la reprise du parent. Cela inclut la correction
des checkpoints de l'etape 12 dans le module partage
`benchmark_parallel_best_of_k.py`, dont les fonctions de calcul restent
identiques. Une modification inconnue des sources, des donnees, des modeles,
des versions ou des budgets reste bloquante ; le message indique les champs
incompatibles. Ce mecanisme ne desactive pas le controle du manifeste.

Le manifeste initial et les checkpoints termines restent inchanges. Leur hash
de reference reste celui du manifeste initial, y compris pour les nouvelles
experiences. Chaque reprise ecrit dans `resumes/` son inventaire actuel et les
corrections compatibles ; les nouveaux checkpoints indiquent aussi le hash de
cet inventaire d'execution. Les sauvegardes du parent sont videes sur disque
avant leur publication atomique, hors budget de decision. Un redemarrage ne
conduit donc pas a refaire les experiences dont le checkpoint est valide.

## Metriques

Chaque fichier est exporte en CSV et JSON :

- `per_scenario` : observations, echecs, timeouts et experiences encore en attente ;
- `summary` : par famille/budget/methode, nombre prevu/termine/faisable/sans slack,
  taux sur les experiences terminees, temps et ecarts aux references MILP longues ;
- `paired_comparisons` : MILP/pipeline, MILP/JEPA, pipeline/JEPA sur les memes
  scenarios reussis, avec effectifs, victoires, egalites et ecarts moyens/medians.

Les agregats sur les seuls succes propres a chaque methode sont nommes
`own_feasible` et ne doivent pas etre compares comme des cohortes identiques.
Les champs `common` utilisent l'intersection des trois methodes, au meme budget.
Cette intersection peut changer avec le budget : les effectifs sont indispensables.
Un tableau en cours distingue toujours les experiences manquantes des echecs.

L'ecart relatif est `100*(cout-reference)/abs(reference)` ; il est indefini si
la reference est nulle. Les references longues conservent leur statut optimal
ou limite de temps. Elles ne sont pas le concurrent MILP sous budget court.
Le slack additionne energie non servie et surproduction evacuee ; ces variables
sont permises et penalisees. Le seuil sans slack est 0,001 MWh.

Les controles de preparation utilisent des candidats, matrices et processus
simules. Ils ne constituent pas une validation numerique d'un solve reel ni
une mesure des performances du GPU ou de HiGHS. Aucun solve/entrainement/inference
n'est lance lors de cette preparation.
