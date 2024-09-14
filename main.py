from create_models import Models
from visualizer import Visualizer


model = Models()
model_visualizer = Visualizer(model.get_average_data("blue"), 60)

model_visualizer.start_animation()
