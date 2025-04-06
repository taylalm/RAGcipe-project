from sentence_transformers import CrossEncoder 
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
model.save('./cross_encoder_model')
