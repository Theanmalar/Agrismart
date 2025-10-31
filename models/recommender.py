def get_recommendations(pred_class, pred_type):
    if pred_type == "soil":
        return [
            f"Best crop for {pred_class} soil: Wheat, Maize",
            "Recommended irrigation: Drip",
            "Organic fertilizer: Compost, Vermicompost",
            "Sowing time: Early Monsoon"
        ]
    else:  # disease
        if pred_class == "Healthy":
            return ["No disease detected", "Maintain regular irrigation", "Use balanced fertilizers"]
        if pred_class == "Blight":
            return ["Spray Mancozeb 0.25%", "Remove affected leaves", "Rotate crops"]
        if pred_class == "Rust":
            return ["Apply Propiconazole 0.1%", "Ensure proper spacing", "Avoid overhead irrigation"]
        return ["Apply recommended fungicide", "Maintain field hygiene"]