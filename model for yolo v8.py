import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
import sqlite3
from datetime import datetime
import os
import logging
from typing import Dict, List, Tuple, Optional
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HydroponicMonitoringSystem:
    """
    AI-powered hydroponic plant monitoring system using YOLOv8
    Detects plant health, pests, diseases, and growth stages
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the monitoring system
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.db_path = "hydroponic_monitoring.db"
        self.setup_database()
        
        # Custom classes for hydroponic monitoring
        self.plant_classes = {
            0: "healthy_plant",
            1: "diseased_plant", 
            2: "pest_infestation",
            3: "nutrient_deficiency",
            4: "seedling",
            5: "mature_plant",
            6: "flowering_stage",
            7: "fruiting_stage",
            8: "aphids",
            9: "whiteflies",
            10: "spider_mites",
            11: "leaf_spot",
            12: "powdery_mildew",
            13: "root_rot"
        }
        
        # Health status mapping
        self.health_status = {
            "healthy_plant": "healthy",
            "diseased_plant": "diseased",
            "pest_infestation": "pest_detected",
            "nutrient_deficiency": "nutrient_issue"
        }
        
        # Growth stage mapping
        self.growth_stages = {
            "seedling": 1,
            "mature_plant": 2,
            "flowering_stage": 3,
            "fruiting_stage": 4
        }
    
    def setup_database(self):
        """Setup SQLite database for storing monitoring data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plant_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                plant_id TEXT,
                detection_type TEXT,
                confidence REAL,
                bbox_x REAL,
                bbox_y REAL,
                bbox_width REAL,
                bbox_height REAL,
                health_status TEXT,
                growth_stage INTEGER,
                recommendations TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                plant_id TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_plants(self, image: np.ndarray) -> List[Dict]:
        """
        Detect plants and their conditions in the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection results
        """
        results = self.model(image, conf=self.confidence_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Get class name
                    class_name = self.plant_classes.get(class_id, "unknown")
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'health_status': self.health_status.get(class_name, "unknown"),
                        'growth_stage': self.growth_stages.get(class_name, 0)
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def analyze_plant_health(self, detections: List[Dict]) -> Dict:
        """
        Analyze overall plant health from detections
        
        Args:
            detections: List of detection results
            
        Returns:
            Health analysis summary
        """
        total_plants = len([d for d in detections if 'plant' in d['class_name']])
        healthy_plants = len([d for d in detections if d['health_status'] == 'healthy'])
        diseased_plants = len([d for d in detections if d['health_status'] == 'diseased'])
        pest_detected = len([d for d in detections if d['health_status'] == 'pest_detected'])
        nutrient_issues = len([d for d in detections if d['health_status'] == 'nutrient_issue'])
        
        health_percentage = (healthy_plants / total_plants * 100) if total_plants > 0 else 0
        
        analysis = {
            'total_plants': total_plants,
            'healthy_plants': healthy_plants,
            'diseased_plants': diseased_plants,
            'pest_detected': pest_detected,
            'nutrient_issues': nutrient_issues,
            'health_percentage': health_percentage,
            'status': self._get_overall_status(health_percentage)
        }
        
        return analysis
    
    def _get_overall_status(self, health_percentage: float) -> str:
        """Get overall system status based on health percentage"""
        if health_percentage >= 90:
            return "excellent"
        elif health_percentage >= 75:
            return "good"
        elif health_percentage >= 50:
            return "moderate"
        else:
            return "critical"
    
    def generate_recommendations(self, detections: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations based on detections
        
        Args:
            detections: List of detection results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for pests
        pests = [d for d in detections if d['class_name'] in ['aphids', 'whiteflies', 'spider_mites']]
        if pests:
            recommendations.append("🐛 Pest detected! Consider biological pest control or neem oil treatment")
        
        # Check for diseases
        diseases = [d for d in detections if d['class_name'] in ['leaf_spot', 'powdery_mildew', 'root_rot']]
        if diseases:
            recommendations.append("🦠 Disease detected! Improve air circulation and check humidity levels")
        
        # Check for nutrient deficiency
        nutrient_issues = [d for d in detections if d['health_status'] == 'nutrient_issue']
        if nutrient_issues:
            recommendations.append("🧪 Nutrient deficiency detected! Check and adjust nutrient solution pH and EC levels")
        
        # Growth stage recommendations
        flowering_plants = [d for d in detections if d['class_name'] == 'flowering_stage']
        if flowering_plants:
            recommendations.append("🌸 Plants in flowering stage! Increase phosphorus and potassium in nutrient solution")
        
        fruiting_plants = [d for d in detections if d['class_name'] == 'fruiting_stage']
        if fruiting_plants:
            recommendations.append("🍅 Plants in fruiting stage! Maintain consistent watering and support heavy branches")
        
        return recommendations
    
    def save_detection_data(self, detections: List[Dict], plant_id: str = None):
        """Save detection data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for detection in detections:
            bbox = detection['bbox']
            recommendations = self.generate_recommendations([detection])
            
            cursor.execute('''
                INSERT INTO plant_detections 
                (plant_id, detection_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, 
                 health_status, growth_stage, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                plant_id or f"plant_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                detection['class_name'],
                detection['confidence'],
                bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1],
                detection['health_status'],
                detection['growth_stage'],
                '; '.join(recommendations)
            ))
        
        conn.commit()
        conn.close()
    
    def create_alert(self, alert_type: str, severity: str, message: str, plant_id: str = None):
        """Create system alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_alerts (alert_type, severity, message, plant_id)
            VALUES (?, ?, ?, ?)
        ''', (alert_type, severity, message, plant_id))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"ALERT: {severity} - {message}")
    
    def process_image(self, image_path: str, plant_id: str = None) -> Dict:
        """
        Process a single image and return complete analysis
        
        Args:
            image_path: Path to the image file
            plant_id: Optional plant identifier
            
        Returns:
            Complete analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect plants
        detections = self.detect_plants(image)
        
        # Analyze health
        health_analysis = self.analyze_plant_health(detections)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(detections)
        
        # Save data
        self.save_detection_data(detections, plant_id)
        
        # Create alerts for critical issues
        if health_analysis['health_percentage'] < 50:
            self.create_alert(
                "health_critical", 
                "high", 
                f"Plant health critical: {health_analysis['health_percentage']:.1f}%",
                plant_id
            )
        
        if health_analysis['pest_detected'] > 0:
            self.create_alert(
                "pest_detected",
                "medium",
                f"Pest detected on {health_analysis['pest_detected']} plants",
                plant_id
            )
        
        # Create annotated image
        annotated_image = self.draw_detections(image, detections)
        
        return {
            'detections': detections,
            'health_analysis': health_analysis,
            'recommendations': recommendations,
            'annotated_image': annotated_image,
            'timestamp': datetime.now().isoformat()
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Annotated image
        """
        img_copy = image.copy()
        
        # Color mapping for different detection types
        colors = {
            'healthy': (0, 255, 0),      # Green
            'diseased': (0, 0, 255),     # Red
            'pest_detected': (255, 0, 0), # Blue
            'nutrient_issue': (0, 255, 255), # Yellow
            'unknown': (128, 128, 128)    # Gray
        }
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color based on health status
            color = colors.get(detection['health_status'], colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img_copy
    
    def process_video_stream(self, video_source: int = 0, output_path: str = None):
        """
        Process live video stream for real-time monitoring
        
        Args:
            video_source: Camera index or video file path
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_source)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30th frame to reduce computational load
                if frame_count % 30 == 0:
                    detections = self.detect_plants(frame)
                    
                    # Create alerts for critical issues
                    for detection in detections:
                        if detection['health_status'] in ['diseased', 'pest_detected']:
                            plant_id = f"stream_plant_{frame_count}"
                            self.create_alert(
                                detection['health_status'],
                                "medium",
                                f"Detected {detection['class_name']} with confidence {detection['confidence']:.2f}",
                                plant_id
                            )
                
                # Draw detections on every frame
                detections = self.detect_plants(frame)
                annotated_frame = self.draw_detections(frame, detections)
                
                # Add system info overlay
                self.add_info_overlay(annotated_frame, detections)
                
                # Display frame
                cv2.imshow('Hydroponic Monitoring', annotated_frame)
                
                # Save frame if output path specified
                if output_path:
                    out.write(annotated_frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
    
    def add_info_overlay(self, image: np.ndarray, detections: List[Dict]):
        """Add information overlay to image"""
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(image, f"Time: {timestamp}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection count
        plant_count = len([d for d in detections if 'plant' in d['class_name']])
        cv2.putText(image, f"Plants detected: {plant_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add health status
        health_analysis = self.analyze_plant_health(detections)
        status_color = {
            'excellent': (0, 255, 0),
            'good': (0, 255, 255),
            'moderate': (0, 165, 255),
            'critical': (0, 0, 255)
        }
        
        color = status_color.get(health_analysis['status'], (255, 255, 255))
        cv2.putText(image, f"Health: {health_analysis['status'].upper()}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def get_monitoring_dashboard(self) -> Dict:
        """Get dashboard data for monitoring system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent detections
        cursor.execute('''
            SELECT * FROM plant_detections 
            ORDER BY timestamp DESC LIMIT 10
        ''')
        recent_detections = cursor.fetchall()
        
        # Get active alerts
        cursor.execute('''
            SELECT * FROM system_alerts 
            WHERE resolved = FALSE 
            ORDER BY timestamp DESC
        ''')
        active_alerts = cursor.fetchall()
        
        # Get health statistics
        cursor.execute('''
            SELECT health_status, COUNT(*) as count
            FROM plant_detections 
            WHERE date(timestamp) = date('now')
            GROUP BY health_status
        ''')
        health_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'recent_detections': recent_detections,
            'active_alerts': active_alerts,
            'health_stats': health_stats,
            'timestamp': datetime.now().isoformat()
        }

# Example usage and integration functions
def main():
    """Example usage of the hydroponic monitoring system"""
    
    # Initialize the system
    monitor = HydroponicMonitoringSystem()
    
    # Example 1: Process a single image
    try:
        image_path = "hydroponic_plants.jpg"  # Replace with actual image path
        if os.path.exists(image_path):
            results = monitor.process_image(image_path, plant_id="greenhouse_1")
            
            print("🌱 Hydroponic Monitoring Results:")
            print(f"Plants detected: {results['health_analysis']['total_plants']}")
            print(f"Health percentage: {results['health_analysis']['health_percentage']:.1f}%")
            print(f"Status: {results['health_analysis']['status']}")
            
            if results['recommendations']:
                print("\n📝 Recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")
            
            # Save annotated image
            cv2.imwrite("annotated_plants.jpg", results['annotated_image'])
            print("✅ Annotated image saved as 'annotated_plants.jpg'")
        
    except Exception as e:
        print(f"Error processing image: {e}")
    
    # Example 2: Start live monitoring (uncomment to use)
    # print("\n🎥 Starting live monitoring...")
    # print("Press 'q' to quit")
    # monitor.process_video_stream(video_source=0)  # Use webcam
    
    # Example 3: Get dashboard data
    dashboard = monitor.get_monitoring_dashboard()
    print(f"\n📊 Dashboard Summary:")
    print(f"Recent detections: {len(dashboard['recent_detections'])}")
    print(f"Active alerts: {len(dashboard['active_alerts'])}")
    print(f"Health statistics: {dashboard['health_stats']}")

if __name__ == "__main__":
    main()