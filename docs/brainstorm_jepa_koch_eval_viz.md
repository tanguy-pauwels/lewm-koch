# Brainstorming: Evaluation et Visualisation du JEPA sur KOCH

## 1) Contexte et objectif

Le run d'entrainement JEPA sur dataset KOCH est fonctionnel. La suite prioritaire est d'expliquer ce que le modele apprend dans son espace latent, malgre une architecture volontairement abstraite.

Etat actuel du repo a prendre en compte:
- Le modele expose `encode`, `predict`, `rollout` dans `src/jepa.py`.
- Le training principal est dans `src/train.py`.
- L'evaluation existante (`src/eval.py`) est orientee planning/policy (`WorldModelPolicy`, `RandomPolicy`), pas introspection latente.
- Il n'y a pas de decodeur image natif dans l'architecture JEPA actuelle.

Objectif de cette note:
- Definir un cadre d'evaluation/visualisation interpretable pour les latents.
- Prioriser une strategie `latent-first`.
- Encadrer la piste "imagination visuelle" comme extension exploratoire et non bloquante.

## 2) Points d'observation disponibles (immediatement exploitables)

Signaux internes accessibles sans changer l'architecture:
- Embeddings encodeur `emb` produits par `encode`.
- Embeddings predits par le predicteur via `predict`/`rollout`.
- Erreur de prediction latente (MSE entre pred et target latent).
- Variables robot du dataset: `action`, `proprio`, `state`.

Signaux d'evaluation utiles:
- Evolution temporelle des trajectoires latentes sur un episode.
- Separation eventuelle entre regimes de comportement (segments de tache, transitions, echec/reussite).
- Distance latent-space entre etats visuellement/physiquement proches.

## 3) Visualisations latentes v1 (focus principal)

### PCA

Avantages:
- Rapide et stable, meme sur volumes importants.
- Projection deterministe (comparaisons entre runs plus propres).
- Bonne lecture de la structure globale et des axes dominants de variance.
- Fournit une base numerique claire via variance expliquee.

Inconvenients:
- Methode lineaire, ne capture pas bien des varietes non lineaires.
- Peut ecraser des structures locales importantes.
- Interprétation parfois difficile si plusieurs facteurs sont entremeles.

Quand l'utiliser:
- Toujours en premier passage pour diagnostic global.
- Pour comparer plusieurs runs/checkpoints avec protocole reproductible.
- Pour initier des analyses temporelles (trajectoires PC1/PC2).

### t-SNE

Avantages:
- Met mieux en evidence les structures locales et clusters non lineaires.
- Souvent plus lisible qualitativement pour detecter groupes de comportements.

Inconvenients:
- Non deterministe sans seed stricte.
- Sensible aux hyperparametres (`perplexity`, `learning_rate`, iterations).
- Les distances globales et la geometrie globale sont peu fiables.
- Plus couteux en calcul sur grands volumes.

Quand l'utiliser:
- En second passage, une fois la structure globale comprise via PCA.
- Pour inspecter des sous-ensembles ciblés (episodes, regimes, erreurs).

### Recommandation pratique

- Utiliser **PCA pour la structure globale**.
- Utiliser **t-SNE pour la structure locale**.
- Ne jamais conclure uniquement a partir de t-SNE sans controles complementaires.

### Risques d'interpretation a expliciter dans chaque analyse

- Projection 2D potentiellement trompeuse vis-a-vis du latent haute dimension.
- Sensibilite de t-SNE a la seed et aux hyperparametres.
- Biais de selection des episodes/fenetres.
- Correlation visuelle ne signifie pas causalite comportementale.

## 4) Protocoles d'analyse proposes

### 4.1 Vue "snapshot" embeddings par episode

But:
- Capturer l'etat latent frame par frame sur episodes echantillonnes.

Protocole:
- Echantillonner `N` episodes de reference (reproductibles via seed).
- Extraire embeddings encodeur sur toutes les frames.
- Projeter en PCA puis t-SNE.
- Colorer les points par temps, episode, et eventuellement score/phase de tache.

Sorties attendues:
- Scatter PCA global.
- Scatter t-SNE local.
- Tableau des stats de projection (variance expliquee PCA, params t-SNE).

### 4.2 Vue "trajectoires temporelles" dans le latent

But:
- Comprendre la dynamique de trajectoire plutot que des points statiques.

Protocole:
- Pour un episode, relier les points de projection dans l'ordre temporel.
- Superposer encode latent vs predicted latent.
- Mesurer la derive latente selon horizon de rollout.

Sorties attendues:
- Courbes de trajectoire 2D par episode.
- Erreur latente par pas de temps.
- Visualisation de zones de divergence.

### 4.3 Vue "voisinages" (nearest neighbors)

But:
- Tester la coherence semantique locale du latent.

Protocole:
- Pour un ancrage latent, recuperer les `k` plus proches voisins.
- Afficher metadata associees: distance latente, temps, action/proprio/state.
- Comparer voisins intra-episode et inter-episode.

Sorties attendues:
- Table NN avec distances.
- Figures illustrant ancre + voisins.
- Statistiques de coherence locale.

### 4.4 Correlation latents <-> variables robot

But:
- Quantifier si le latent code des informations physiques utiles.

Protocole:
- Probing lineaire/regression sur `action`, `proprio`, `state`.
- Evaluer R2/MSE (regression) ou metriques de classification si discretisation.
- Comparer encode latents vs predicted latents.

Sorties attendues:
- `probe_metrics.json` avec metriques par variable.
- Graphiques des performances par signal.

## 5) Piste decodage/imagination (exploratoire)

Pourquoi ce n'est pas direct:
- Le JEPA courant n'apprend pas de reconstruction image.
- L'objectif est prediction dans un espace latent, pas generation de pixels.

Option 1: latent-only (prioritaire court terme)
- Ne pas decoder en pixel.
- Visualiser et evaluer uniquement l'espace latent et sa dynamique.
- Avantage: zero changement architecture, faible risque.

Option 2: petit head de reconstruction
- Ajouter un decodeur leger latent -> image.
- Entrainer en multi-objectifs avec perte de reconstruction.
- Risque: perturber l'apprentissage JEPA, cout d'integration.

Option 3: decodeur generatif dedie
- Geler JEPA, entrainer un modele separé conditionne par latents.
- Plus flexible et potentiellement qualitatif.
- Cout de recherche/compute plus eleve et pipeline plus complexe.

Criteres de decision pour lancer un decodeur:
- Le diagnostic latent-first atteint un plateau d'interpretabilite.
- Les questions produit/recherche exigent explicitement des rendus pixel.
- Budget compute et calendrier permettent une piste additionnelle.

## 6) Roadmap experimentale

### Phase A: diagnostics latents non-invasifs

Livrables:
- Extractions embeddings encode/predict sur sous-ensemble KOCH.
- Projections PCA+t-SNE.
- Trajectoires temporelles.

Critere de sortie:
- Vue claire de la structure globale et locale du latent.

### Phase B: probing quantitatif

Livrables:
- Probes sur `action`, `proprio`, `state`.
- Mesures d'erreur de prediction latente selon horizon.

Critere de sortie:
- Evidence quantifiee sur le contenu informationnel des latents.

### Phase C: decodage exploratoire (optionnel)

Livrables:
- Prototype de l'option choisie (head leger ou decodeur dedie).
- Evaluation qualitative et quantitative de la fidelite image.

Critere de sortie:
- Decision go/no-go sur integration durable du decoding.

## 7) Backlog concret (sans implementation immediate)

Backlog priorise:
1. Script extraction latents `encode`/`predict` sur episodes KOCH.
2. Script projection PCA + export CSV.
3. Script projection t-SNE + export CSV.
4. Script plots trajectoires temporelles.
5. Script nearest-neighbors latents.
6. Script probing lineaire sur `action`, `proprio`, `state`.
7. Script rapport agregeant metriques + figures.

## 8) Contrat d'artefacts d'evaluation (mini-spec)

Les futures implementations devront produire au minimum:

- `eval_artifacts/<run_id>/latent_pca.csv`
- `eval_artifacts/<run_id>/latent_tsne.csv`
- `eval_artifacts/<run_id>/probe_metrics.json`
- `eval_artifacts/<run_id>/figures/*.png`

Contraintes minimales:
- `run_id` doit correspondre au sous-dossier de run (`RUN_SUBDIR`).
- Les CSV doivent inclure des identifiants de contexte (`episode_idx`, `step_idx`) et les coordonnees projetees.
- Le JSON de probing doit etre lisible machine et comparer plusieurs signaux robot.

## 9) Hypotheses et decisions prises

- Format: note de recherche orientee exploration.
- Priorite v1: structure de l'espace latent.
- Strategie retenue: `latent-first`.
- Le decoding image est maintenu en piste exploratoire secondaire, non bloquante.
