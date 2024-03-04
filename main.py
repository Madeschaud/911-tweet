from Preprocessing.preprocessor import preprocessor_all
from colorama import Fore
from Add_data.new_data import download_data

def main():
    print(Fore.RED + 'Main' + Fore.WHITE)

    preprocessor_all()
