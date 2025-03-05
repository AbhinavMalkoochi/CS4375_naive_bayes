from bow import Bow
from bernoulli import Bernoulli
def main():

    bow = Bow()
    bow.test_bow()
    bow.test_mcap_logistic_regression()
    print("-------------------------------")
    print("Bernoulli")
    bernoulli = Bernoulli()
    bernoulli.test_bernoulli()
    bernoulli.test_mcap_logistic_regression()
if __name__ == "__main__":
    main()


