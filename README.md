# Kursinio darbo projektas: „Baltymų sekų klasifikacija naudojant giliuosius neuroninius tinklus“

**Atliko:** Laura Černyškovaitė  

Šiame projekte sukurtas giluminio neuroninio tinklo modelis, skirtas baltymų sekų klasifikacijai į:  
- **0** – nefermentai  
- **1** – fermentai  

---
## Turinys
- [Kas yra faile?](#kas-yra-faile)
- [Reikalavimai](#reikalavimai)
- [Naudojimas](#naudojimas)
- [Atnaujinimai](#atnaujinimai)
  - [v2.0 (2026-01)](#v20-2026-01)
  - [Kaip paleisti?](#kaip-paleisti)
  - [Kas sukuriama paleidus koda?](#kas-sukuriama-paleidus-koda)


---

## Kas yra faile?
- `enzyme_vs_nonenzyme.ipynb` – visas kodas (Google Colab / Jupyter).  
- `ncbi_enzyme_dataset.csv` – duomenų failas (sekos + etiketės).  
- `images/` – mokymo grafikai ir klaidų matrica.  
- `enzyme_ncbi_model.keras` – išsaugotas modelis.
- `enzyme_nonenzyme.py` – v2.0 kodas

---

## Reikalavimai

Norint paleisti projektą, reikia įdiegti šias bibliotekas:  

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```
---

## Naudojimas

1. Atsidarykite `enzyme_vs_nonenzyme.ipynb` Google Colab arba Jupyter Notebook aplinkoje.  
2. Įkelkite duomenų failą `ncbi_enzyme_dataset.csv` į tą pačią direktoriją.  
3. Paleiskite visą kodą.  
4. Rezultatai (tikslumas, AUC, klaidų matrica ir grafikai) bus sugeneruoti aplanke `images/`.  
5. Išsaugotas modelis bus sukurtas faile `enzyme_ncbi_model.keras`.  
6. Norint atlikti pavyzdinę prognozę, galima naudoti:  
   ```python
   predict_protein_sequence("MKKLIALKHKDEMKKLAAAGGGSSSSVVVVVVNNNPPPQQQ")

---
## Atnaujinimai

### v2.0 (2026-01)
Įkelta atnaujinta projekto versija su patobulinta duomenų paruošimo ir modeliavimo eiga.

**Kas naujo:**
- Pridėtas `enzyme_nonenzyme.py` (alternatyva notebook'ui) – pilnas paleidžiamas kodas.
- Įgyvendintas duomenų išplėtimas (angl. *data augmentation*) – duomenų kiekis padidintas nuo 3941 iki 7882 sekų.
- Pridėtos 5 biocheminės savybės: `length`, `molecular_weight`, `isoelectric_point`, `gravy`, `aromaticity`.
- Sukurtas multimodalus modelis (sekos + biocheminiai požymiai): CNN + BiLSTM + požymių šaka.
- Pridėtas sujungtas (angl. *ensemble / stacking*) modelis: neuroninio modelio tikimybė + biocheminiai požymiai → Random Forest.
- Automatiškai generuojami ir išsaugomi grafikai (`results_enhanced/images/`): mokymo kreivės, ROC, PR, klaidų matrica, požymių svarba, modelių palyginimas.
- Eksportuojamos prognozės ir ataskaitos (`results_enhanced/reports/`): `all_predictions.csv`, klaidų failai, `feature_importance.csv`, `final_report.json`.

### Kaip paleisti?
```bash
!git clone https://github.com/LauraCernyskovaite/enzyme-vs-nonenzyme.git
!cd enzyme-vs-nonenzyme
!pip -q install tensorflow scikit-learn pandas numpy matplotlib biopython
!python enzyme_nonenzyme.py
```

### Kas sukuriama paleidus koda?

Paleidus `enzyme_nonenzyme.py`, automatiškai sukuriamas aplankas `results_enhanced/` su tokia struktūra:

- `results_enhanced/augmented_data/` – išsaugotas išplėstas (angl. *augmented*) duomenų rinkinys  
  (pvz., `augmented_dataset.csv`).
- `results_enhanced/images/` – sugeneruoti grafikai (mokymo kreivės, ROC, PR, klaidų matricos, požymių svarba, modelių palyginimas).
- `results_enhanced/models/` – išsaugotas geriausias modelis  
  (pvz., `best_multimodal.keras`).
- `results_enhanced/reports/` – ataskaitos ir prognozės failuose  
  (pvz., `all_predictions.csv`, `errors_multimodal.csv`, `errors_ensemble.csv`, `feature_importance.csv`, `final_report.json`).

> Pastaba: šie aplankai ir failai sugeneruojami automatiškai vykdymo metu, todėl jų nereikia kurti rankiniu būdu. Colab aplinkos failai yra laikini (dingsta išjungus sesiją). Jei reikia išsisaugoti rezultatus, rekomenduojama prijungti Google Drive ir results_enhanced/ katalogą nukopijuoti į Drive
