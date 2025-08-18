# Iris Flower Classification API

This API classifies iris flowers into three species based on four measurements.

## Model Details
- Algorithm: RandomForestClassifier
- Features: sepal length, sepal width, petal length, petal width
- Target: species (setosa, versicolor, virginica)

## API Endpoints

### Health Check
- `GET /` - Returns API status

### Prediction
- `POST /predict` - Accepts flower measurements and returns predicted species
  - Example request:
    ```json
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    ```
  - Example response:
    ```json
    {
        "species": "setosa",
        "species_id": 0,
        "probabilities": {
            "setosa": 0.9,
            "versicolor": 0.1,
            "virginica": 0.0
        },
        "confidence": 0.9
    }
    ```

### Model Information
- `GET /model-info` - Returns model metadata and feature importance

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python train_model.py`
3. Run API: `uvicorn main:app --reload`
4. Access docs at: http://localhost:8000/docs