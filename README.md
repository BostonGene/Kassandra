 
![Alt text](images/kassandra_logo.png?raw=true "Title")
 
https://science.bostongene.com/kassandra/
 
<!-- TABLE OF CONTENTS -->
<details open="open">
 <summary>Table of Contents</summary>
 <ol>
   <li>
     <a href="#about-the-project">About The Project</a>
   </li>
   <li>
     <a href="#getting-started">Getting Started</a>
       <ul>
           <li><a href="#Data">Data</a></li>
       </ul>
       <ul>
           <li><a href="#Model training">Model training</a></li>
       </ul>
       <ul>
           <li><a href="#Transcripts to genes">Transcripts to genes</a></li>
       </ul>
   </li>
   <li>
     <a href="#publication">Publication</a>
   </li>
   <li>
     <a href="#license">License</a>
   </li>
 </ol>
</details>
 
 
<!-- ABOUT THE PROJECT -->
## About The Project
 
Kassandra, a robust and accurate cell deconvolution tool developed for analysis of healthy tissue and tumor biopsies. Based on RNA-seq NGS data of a biological sample, Kassandra predicts cellular composition including stromal and immune elements by analyzing the gene expression. This will lead to an improved understanding of the tumor microenvironment, which is a critical factor in cancer pathogenesis, clinical outcome, and therapeutic resistance.
![Alt text](images/abstract.png?raw=true "Title")
 
Kassandra is a decision tree machine learning-algorithm trained on a collection of over thousands of RNA profiles from various sorted cell types. Performance was validated on over 4,000 H&E tissue slides and more than 1,000 samples comprising normal and tumor tissues by comparison with cytometric, immunohistochemical or single-cell RNA sequencing measurements of the same tissue.
 
<!-- Getting Started -->
## Getting Started
 
"Model Training.ipynb” provides limited example of training blood deconvolution model. The model is relatively small and can be trained on an average laptop. To reproduce the full publication model use "configs/full_blood_model.yaml" config. It requires 40-60gb of memory for generation of training data. adsf
 
<!-- Data -->
## Data
The example dataset consists of 4 tables in the 'data' folder, two with annotation and two with expressions of cancer cell lines and sorted cells samples. To train full model download data from https://science.bostongene.com/kassandra/downloads.
 
 
<!-- Transcripts to genes -->
## Transcripts to genes
Kassandra deconvolution is trained and can be used on gene TPM expressions obtained by the specific procedure described in the publication methods section. In short there are transcripts and genes that are removed from expressions. Tumor and blood models use slightly different procedures. Both procedures demonstrated in "Model Training.ipynb" notebook.
 
![Alt text](images/tr_to_genes.jpg?raw=true "Title")
 
Inputs in form of gene TPM produced by other tools or procedures different from described above are not recommended.
 
 
<!-- publication -->
## Publication
 
### <b> Precise reconstruction of the tumor microenvironment using bulk RNA-seq and a unique machine learning-based algorithm trained on artificial transcriptomes </b>
 
Aleksandr Zaitsev1, Maksim Chelushkin1, Daniiar Dyikanov1, Ilya Cheremushkin1, Boris Shpak1, Krystle Nomie1, Vladimir Zyrin1, Katerina Nuzhdina1, Yaroslav Lozinsky1, Anastasia Zotova1 , Sandrine Degryse1, Nikita Kotlov1, Arthur Baisangurov1, Vladimir Shatsky1, Daria Afenteva1, Susan Raju Paul2, Diane L. Davies3, Patrick M. Reeves2, Michael Lanuti3, Michael F. Goldberg1, Cagdas Tazearslan1, Madison Chasse1, Iris Wang1, Mary Abdou1, Sharon M. Aslanian1, Samuel Andrewes1, James J. Hsieh4, Akshaya Ramachandran4, Yang Lyu4, Ilia Galkin1, Viktor Svekolkin1, Leandro Cerchietti5, Mark C. Poznansky2, Ravshan Ataullakhanov1, Nathan Fowler1,6*, Alexander Bagaev1*
 
##### 1BostonGene, Corp, Waltham, MA, USA
##### 2The Vaccine and Immunotherapy Center, Massachusetts General Hospital, Boston, MA, USA
##### 3Division of Thoracic Surgery, Massachusetts General Hospital, Boston, MA, USA
##### 4Molecular Oncology, Division of Oncology, Department of Medicine, Washington University, St. Louis, MO, USA
##### 5Division of Hematology and Medical Oncology, Weill Cornell Medicine, New York City, NY, USA
##### 6Department of Lymphoma and Myeloma, MD Anderson Cancer Center, Houston, TX, USA
 
<!-- Licencse -->
## Licencse
 
BY UTILIZING THE CODE, YOU ARE CONSENTING TO BE AND AGREE TO BE BOUND BY ALL OF THE TERMS OF THIS LIMITED LICENSE, SEE "LICENSE.txt"
 

