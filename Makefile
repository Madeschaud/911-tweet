run_preprocess:
	@python -c 'from Preprocessing.preprocessor import cleaning; cleaning()'

run_main:
	@python -c 'from main import main; main()'
