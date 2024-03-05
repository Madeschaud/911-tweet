run_preprocess:
	@python -c 'from Preprocessing.preprocessor import preprocessor_all; preprocessor_all()'

run_main:
	@python -c 'from main import main; main()'

install_dep:
	pip install -r requirements.txt

save_model:
  @python -c ‘from registry import save_model; save_model()’
