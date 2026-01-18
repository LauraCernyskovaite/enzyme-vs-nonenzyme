# Kursinio darbo projektas: „Baltymų sekų klasifikacija naudojant giliuosius neuroninius tinklus“

**Atliko:** Laura Černyškovaitė  

Šiame projekte sukurtas giluminio neuroninio tinklo modelis, skirtas baltymų sekų klasifikacijai į:  
- **0** – nefermentai  
- **1** – fermentai  

Modelis pagrįstas CNN ir BiLSTM architektūrų deriniu, optimizuotas su Adam algoritmu, vertintas pagal tikslumą, ROC-AUC ir klaidų matricą.

---

## Turinys
- [Kas yra faile](#kas-yra-faile)  
- [Reikalavimai](#reikalavimai)  
- [Naudojimas](#naudojimas)  

---

## Kas yra faile?
- `enzyme_vs_nonenzyme.ipynb` – visas kodas (Google Colab / Jupyter).  
- `ncbi_enzyme_dataset.csv` – duomenų failas (sekos + etiketės).  
- `images/` – mokymo grafikai ir klaidų matrica.  
- `enzyme_ncbi_model.keras` – išsaugotas modelis.  

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

**Pagrindiniai rezultatai (testavimo rinkinyje):**
- Multi-Modal: accuracy ≈ 97.30%, ROC-AUC ≈ 0.9952
- Ensemble: accuracy ≈ 98.39%, ROC-AUC ≈ 0.9949
- Klaidų sumažėjo nuo 32 iki 19.


