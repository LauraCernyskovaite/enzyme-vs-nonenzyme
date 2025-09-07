# Kursinis darbas: „Baltymų sekų klasifikacija naudojant giliuosius neuroninius tinklus“

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
