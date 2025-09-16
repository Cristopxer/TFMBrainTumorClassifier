import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GradCAMConfig:
    """Configuration class for Grad-CAM parameters"""
    img_size: Tuple[int, int] = (224, 224)
    alpha: float = 0.4
    last_conv_layer_name: str = 'gradcam_conv'
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


class GradCAMVisualizer:
    """Class for generating and visualizing Grad-CAM heatmaps"""
    
    def __init__(self, model: str, config: Optional[GradCAMConfig] = None):
        """
        Initialize the Grad-CAM visualizer.
        
        Args:
            model: path to the trained TensorFlow model
            config: Configuration parameters for Grad-CAM
        """
        self.model = tf.keras.models.load_model(model)
        self.config = GradCAMConfig()
        self._validate_model()
        
    def _validate_model(self) -> None:
        """Validate that the model has the required convolutional layer"""
        try:
            self.model.get_layer(self.config.last_conv_layer_name)
        except ValueError:
            logger.warning(f"Layer '{self.config.last_conv_layer_name}' not found in model. "
                          "This may cause issues during heatmap generation.")
    
    def _build_grad_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Build the gradient model for Grad-CAM.
        
        Returns:
            Tuple of (convolutional model, classifier model)
        """
        # Create model that maps input to last convolutional layer outputs
        grad_model_conv = tf.keras.models.Model(
            self.model.inputs, 
            self.model.get_layer(self.config.last_conv_layer_name).output
        )
        
        # Find the index of the last convolutional layer
        last_conv_layer_index = None
        for i, layer in enumerate(self.model.layers):
            if layer.name == self.config.last_conv_layer_name:
                last_conv_layer_index = i
                break
        
        if last_conv_layer_index is None:
            raise ValueError(f"Layer '{self.config.last_conv_layer_name}' not found in the model.")
        
        # Build classifier model (layers after the convolutional layer)
        classifier_input = tf.keras.Input(shape=grad_model_conv.output.shape[1:])
        x = classifier_input
        
        # Add layers after the convolutional layer
        for layer in self.model.layers[last_conv_layer_index + 1:]:
            # Only include relevant classification layers
            if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D,
                                tf.keras.layers.Dense,
                                tf.keras.layers.Dropout,
                                tf.keras.layers.BatchNormalization)):
                try:
                    x = layer(x)
                except Exception as e:
                    logger.warning(f"Could not add layer {layer.name} to classifier model: {e}")
        
        classifier_model = tf.keras.models.Model(classifier_input, x)
        
        return grad_model_conv, classifier_model
    
    def _preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.config.img_size)
        img_array = np.asarray(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def generate_heatmap(self, img_array: np.ndarray, top_pred_index: int) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM heatmap for a given image.
        
        Args:
            img_array: Preprocessed image array
            top_pred_index: Index of the top predicted class
            
        Returns:
            Grad-CAM heatmap or None if generation fails
        """
        try:
            grad_model_conv, classifier_model = self._build_grad_model()
            
            with tf.GradientTape() as tape:
                last_conv_layer_output = grad_model_conv(img_array)
                tape.watch(last_conv_layer_output)
                
                preds = classifier_model(last_conv_layer_output)
                
                # Handle different model output formats
                if isinstance(preds, list):
                    predictions_tensor = preds[1] 
                else:
                    predictions_tensor = preds
                
                top_class_channel = predictions_tensor[:, top_pred_index]
            
            # Compute gradients
            grads = tape.gradient(top_class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Generate heatmap
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Apply ReLU and normalize
            heatmap = tf.maximum(heatmap, 0)
            max_heatmap = tf.reduce_max(heatmap)
            
            if max_heatmap != 0:
                heatmap /= max_heatmap
            
            return heatmap.numpy()
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return None
    
    def superimpose_heatmap(self, img_path: str, heatmap: np.ndarray) -> np.ndarray:
        """
        Superimpose heatmap on the original image.
        
        Args:
            img_path: Path to the original image
            heatmap: Grad-CAM heatmap
            
        Returns:
            Image with superimposed heatmap
        """
        # Load and resize original image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.config.img_size)
        img_array = np.asarray(img).astype(np.uint8)
        
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Apply colormap to heatmap
        heatmap_colored = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        superimposed_img = cv2.addWeighted(
            img_array.astype(np.uint8), 0.6, 
            heatmap_colored.astype(np.uint8), 0.4, 0
        )
        
        return superimposed_img
    
    def predict(self, img_array: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Get model predictions for an image.
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Tuple of (predicted class index, prediction scores)
        """
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        
        return predicted_class_index, predictions[0]
    
    def visualize(self, img_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate and visualize Grad-CAM results.
        
        Args:
            img_path: Path to the input image
            output_path: Path to save the visualization
            
        Returns:
            Dictionary containing visualization results
        """
        # Preprocess image
        img_array = self._preprocess_image(img_path)
        
        # Get predictions
        predicted_class_index, prediction_scores = self.predict(img_array)
        predicted_class_name = self.config.class_names[predicted_class_index]
        
        # Generate heatmap
        heatmap = self.generate_heatmap(img_array, predicted_class_index)
        
        if heatmap is None:
            logger.error("Failed to generate heatmap")
            return {}
        
        # Superimpose heatmap
        superimposed_img = self.superimpose_heatmap(img_path, heatmap)
        
        # Load original image for display
        original_img = Image.open(img_path).convert('RGB')
        original_img = original_img.resize(self.config.img_size)
        original_img_array = np.asarray(original_img).astype(np.uint8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        
        axes[0].imshow(original_img_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(superimposed_img)
        axes[2].set_title("Superimposed Heatmap")
        axes[2].axis('off')
        
        # Add prediction text
        prediction_text = "Predictions:\n"
        for i, score in enumerate(prediction_scores):
            prediction_text += f"{self.config.class_names[i]}: {score:.4f} "
        prediction_text += f"\nPredicted Class: {predicted_class_name}"
        
        plt.figtext(0.5, 0.02, prediction_text, ha="center", fontsize=12, 
                   bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
        
        plt.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.15)
        
        # Save visualization if output path is provided
        if output_path:
            plt.savefig(output_path)
        
        # Log predictions
        logger.info("\nPredictions:")
        for i, score in enumerate(prediction_scores):
            logger.info(f"{self.config.class_names[i]}: {score:.4f}")
        logger.info(f"Predicted Class: {predicted_class_name}")
        
        return {
            'original_image': original_img_array,
            'heatmap': heatmap,
            'superimposed_image': superimposed_img,
            'predictions': prediction_scores,
            'predicted_class': predicted_class_index,
            'predicted_class_name': predicted_class_name
        }