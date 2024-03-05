run_preprocess:
	@python -c 'from Preprocessing.preprocessor import preprocessor_all; preprocessor_all()'

run_main:
	@python -c 'from main import main; main()'

run_vectorizer:
	@python -c 'from Preprocessing.vectorizer import vectorizer; vectorizer()'

save_model:
	@python -c 'from registry import save_model; save_model()'
	
install_dep:
	pip install -r requirements.txt
