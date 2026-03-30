<a id="readme-top"></a>
# 🪚 Linear Regression From Scratch - [Project_Theme]

Coding a Linear Regression Algorithm without any ML libraru such as Scikit-Learn.


## 🚀 Table of Contents

1. [General Informations](#-general-informations)
2. [Objectives](#-objectives)
3. [Dataset Description](#-dataset-description)
4. [Methodological Approach](#-methodological-approach)
5. [Project's Structure](#-projects-structure)
6. [Installation and Usage](#-installation-and-usage)
7. [Environnement and Tools](#-environnement-and-tools)
8. [Contact](#-contact)
0. [License](#-license)

## 📋 General Informations

This project is an implement of a linear regression algorithm with a stochastic gradient descent optimizer.

    Modèle de Fichier RDG README --- Général --- Version: 0.1 (2022-11-22)

    Ce fichier README a été généré le [YYYY-MM-DD] par [NAME].

    Dernière mise-à-jour le : [YYYY-MM-DD].

## 🎯 Objectives

* Objective 1: Showcase the implementation of linear regression algorithm without any ML library.
* Objective 2: Explore the mathematical theory behind the famous linear regression algorithm
* Objective 3:
* Objective 4: Deepen under the hood of the Linear Regression algorithm as well as my understanding of Machine Learning algorithms on a technical level (Under the hood)

## 📊 Dataset Description
- **Source**: (Kaggle / Internal / API / etc.) - 🔗 [Lien des données]
- **Size**: XXX rows x XXX columns
- **Target**: `target_column`
- **Features**: Numerical, Categorical, Time-series, etc.


#### Features Description :
| Features     | Data Type      | Description                   |
|--------------|----------------|-------------------------------|
| `feature_1`  | Numerical      | Description of feature 1      |
| `feature_2`  | Categorical    | Description of feature 2      |
| `feature_3`  | Numerical      | Description of feature 3      |
| `feature_4`  | Categorical    | Description of feature 4      |
| ...          | ...            | ...                           |

## 🔎 Methodological Approach

#### 1 - Data Preprocessing
* Missing value treatment
* Encoding categorical variables
* Feature scaling
* Outlier detection

#### 2 - Brief Exploratory Data Analysis
* Correlation heatmaps
* Distribution plots
* Feature importance insights

#### 3 - Model Selection
* Logistic Regression
* Random Forest
* XGBoost

#### 4 - Evaluation Metrics
* Accuracy
* Precision / Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

#### 5 - Fine Tuning
* GridSearch
* RandomSearch


## 📁 Project's Structure

#### Arborescence du projet :
```bash
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── pipeline.py
│   └── evaluate.py
├── models/
│   ├── model.pkl
│   └── best_model.pkl
├── app/
├── requirements.txt
└── README.md
```

## 🧰 Installation and Usage

1. Clonnage du repo
```bash
git clone https://github.com/yourusername/project-name.git
```

2. Navigation dans le project
```bash
cd project-name
```

3. Creation de l'environnement virtuel
```bash
python -m venv venv
```

4. Activer l'environment
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

5. Installation des dépendences
```bash
pip install -r requirements.txt
```

---

### Utilisation du modèle
1. Entraînement
```bash
python src/train.py --config config.yaml
```

2. Evaluation
```bash
python src/evaluate.py --model models/best_model.pth
```

3. Prediction
```bash
python src/predict.py --input sample.csv
```

4. API
```bash
uvicorn app.main:app --reload
```

## 🌱 Environnement and Tools

* Système : MacOS
* Shell : Bash
* Environnement : venv
* Langage : Python 3.12
* Versioning : Git / GitHub
* Data Tools : Numpy, Pandas, Matplotlib, Seaborn, Plotly
* Notebook : Jupyter
* ML Frameworks : Scikit-Learn, XGBoost
* DL Frameworks : TensorFlow / Keras, Pytorch
* Framework API : FastAPI
* Containerization : Docker
* Cloud Platform : AWS
* Orchestration : Airflow, Prefect
* Monitoring : MLflow, Weights & Biases


<!-- ## 📷 Screenshots
-->


## 📬 Contact

|🖋️ Author           | Nicolas Maréchal               |
|------------------|---------------------------------|
|✉️ Mail           | marechal.n@hotmail.com           |
|💼 Project's Link | [![repo-shield]][repo-url]       |
|🔗 Profiles Links | [![github-shield]][github-url] [![linkedin-shield]][linkedin-url]  ![Portfolio][portfolio-shield] [![kaggle-shield]][kaggle-url]|




## 📜 License
This project is licensed under the [...] License - see the `LICENSE.md` file for details.

<p align="right"><a href="#readme-top">- back to top -</a></p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[github-shield]: https://img.shields.io/badge/-Github-353538?style=for-the-badge&logo=Github&logoColor=white
[github-url]: https://github.com/NkL-M

[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logoColor=white
[linkedin-url]: https://linkedin.com/in/nicolas-marechal-oc

[portfolio-shield]: https://img.shields.io/badge/Portfolio-8F1F07?style=for-the-badge&logoColor=white
[portfolio-url]: https://

[kaggle-shield]: https://img.shields.io/badge/Kaggle-353538?style=for-the-badge&logo=Kaggle&logoColor=0077B5
[kaggle-url]: https://www.kaggle.com/nkl974

[repo-shield]: https://img.shields.io/badge/Repository-Project_Example_01-353538?style=for-the-badge&logo=git&logoColor=F14E32
[repo-url]: https://github.com/NkL-M


<!-- ## Remerciements/Inspirations
## 💼 Sources/Remerciements/Inspirations

List of helpful resources to give credit to.

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)
-->
