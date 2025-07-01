import numpy as np
from preprocessing import preprocess_image
from config import CLASSES


def predict_batch(images, model, model_type):
        results = []
        
        for i, image in enumerate(images):
            try:
                # Preprocessing
                processed_image = preprocess_image(image, model_type)
                
                # Prediksi
                prediction = model.predict(processed_image, verbose=0)
                
                # Ambil hasil
                predicted_class = CLASSES[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                results.append({
                    'filename': image.name,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'normal_prob': prediction[0][0] * 100,
                    'aktif_prob': prediction[0][1] * 100,
                    'laten_prob': prediction[0][2] * 100
                })
                
            except Exception as e:
                results.append({
                    'filename': image.name,
                    'predicted_class': 'Error',
                    'confidence': 0,
                    'normal_prob': 0,
                    'aktif_prob': 0,
                    'laten_prob': 0,
                    'error': str(e)
                })
        
        return results