from models import CNN

pos_labels=['Prolongation', 'Repetition', 'Block']

model = CNN(pos_labels)

model.keras_model.summary()
