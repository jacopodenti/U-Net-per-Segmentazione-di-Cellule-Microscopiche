# U-Net per Segmentazione di Cellule Microscopiche

Questo progetto implementa una rete **U-Net** per la **segmentazione di immagini microscopiche** di cellule.  
L’obiettivo è addestrare un modello in grado di distinguere correttamente ogni cellula dal background, generando una maschera segmentata a partire dall’immagine originale.

Il lavoro è interamente documentato nel notebook:

- `unet_cell_segmentation.ipynb`

che contiene:

- esplorazione del dataset,
- definizione del modello U-Net,
- training,
- valutazione,
- esempi di segmentazione (input / maschera / predizione).

---

## 🧪 Dataset

Il dataset originale comprendeva:

-  immagini microscopiche (TIFF/TIF/PNG)
-  maschere di segmentazione corrispondenti

Per motivi di spazio e licenza il dataset **non è incluso** nella repository.  
Tuttavia, nel notebook sono mostrati diversi esempi di:

- immagini di input,
- maschere ground truth,
- output del modello.

Questo permette di valutare chiaramente il comportamento del modello anche senza rieseguire tutto il training.

---

## 🧠 Modello e training

Nel notebook sono presenti tutti i dettagli su:

- architettura U-Net utilizzata,
- loss function,
- ottimizzatore,
- iperparametri,
- metriche (es. Dice, IoU),
- visualizzazione dei risultati finali.

---

## 🚀 Obiettivi raggiunti

In questo progetto ho implementato:

- una pipeline completa di image segmentation in ambito biomedico,
- un modello U-Net funzionante,
- addestramento e valutazione su un dataset reale di immagini di cellule,
- visualizzazione qualitativa e quantitativa dei risultati.
