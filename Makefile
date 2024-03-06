run_preprocess:
	@python -c 'from Preprocessing.preprocessor import preprocessor_all; preprocessor_all()'

run_main:
	@python -c 'from main import main; main()'

run_vectorizer:
	@python -c 'from Preprocessing.vectorizer import vectorizer; vectorizer()'

run_baseline:
	@python -c 'from Model.baseline import baseline_naive; baseline_naive()'

boost_naive_base:
	@python -c 'from Model.boost_naive_base import boost_naive_base; boost_naive_base()'
model_lstm:
	@python -c 'from Model.lstm import model_lstm; model_lstm()'

run_simple_gru:
	@python -c 'from Model.simple_gru import GRU_model; GRU_model()'

xg_boost:
	@python -c 'from Model.xg_boost import xg_boost; xg_boost()'

save_model:
	@python -c 'from registry import save_model; save_model()'

install_dep:
	pip install -r requirements.txt

save_model:
	@python -c 'from registry import save_model, save_results; save_model(), save_results()'

load_model:
	@python -c 'from registry import load_model; load_model()'
