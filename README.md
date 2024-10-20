# Fight Action Prediction

This project focuses on predicting fight actions using deep learning models. We classify various fight-related actions using models like **ResNet152** and **EfficientNetB7**. The actions being predicted include:

- **0**: Hit
- **1**: Kick
- **2**: Punch
- **3**: Push
- **4**: Ride Horse
- **5**: Shoot Gun
- **6**: Stand
- **7**: Wave

[!image](./image.png)

## Models
We trained and fine-tuned the following models:
- **ResNet152**: Fine-tuned on the action dataset.
- **EfficientNetB7**: Another deep learning model used for comparison.

## Dataset
The dataset for training includes labeled images corresponding to the actions mentioned above. Ensure your dataset follows the directory structure:

```
data/
│
├── train/
│   └── <class_name>/
│       └── image1.jpg
├── test/
    └── <class_name>/
        └── image2.jpg
```

Each class folder contains relevant images for that action.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/alihassanml/Fight-Prediction-Project.git
cd Fight-Prediction-Project
pip install -r requirements.txt
```

## Training

To train the models, run the following command:

```bash
python train.py
```

This will start training the model on the dataset located in the `data/train` directory.

## Streamlit Application

You can interact with the trained models using a web interface powered by Streamlit. To run the app:

```bash
streamlit run streamlit_app.py
```

Upload an image, select the model you want to use, and get predictions for the action being performed in the image.

## Results

After training, models will be saved as `.h5` files, and the model performance can be assessed using validation data and metrics such as accuracy, confusion matrix, and classification report.

## Fine-Tuning

Fine-tuning is done by unfreezing the last few layers of the pre-trained models and retraining them with a smaller learning rate to improve accuracy.

## License

This project is licensed under the MIT License.
