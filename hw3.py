import numpy as np
from typing import Union

def poisson_log_pmf(k: Union[int, np.ndarray], rate: float) -> Union[float, np.ndarray]:
    """
    k: A discrete instance or an array of discrete instances
    rate: poisson rate parameter (lambda)

    return the log pmf value for instances k given the rate
    """

    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    k = np.asarray(k)
    if k.ndim == 0:
        log_k_fact = np.sum(np.log(np.arange(1, k + 1))) if k > 0 else 0
    else:
        log_k_fact = np.array([np.sum(np.log(np.arange(1, every_k + 1))) if every_k > 0 else 0 for every_k in k])

    log_p = k * np.log(rate) - rate - log_k_fact
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean = np.mean(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    lower_bound = None
    upper_bound = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    z = norm.ppf(1 - alpha / 2)
    std_err = np.sqrt(lambda_mle / n)
    lower_bound = lambda_mle - std_err * z
    upper_bound = lambda_mle + std_err * z
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    samples = np.asarray(samples)
    rates = np.asarray(rates)
    num_rates = len(rates)
    log_factor = np.array([np.sum(np.log(np.arange(1, x + 1))) if x > 0 else 0 for x in samples])
    likelihoods = np.zeros(num_rates)
    for i, rate in enumerate(rates):
        log_p = samples * np.log(rate) - rate - log_factor
        likelihoods[i] = np.sum(log_p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

class conditional_independence():

    def __init__(self):

        self.X = {0: 0.3, 1: 0.7}
        self.Y = {0: 0.3, 1: 0.7}
        self.C = {0: 0.5, 1: 0.5}

        self.X_Y = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.2,
            (1, 1): 0.5
        }

        self.X_C = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.4,
            (1, 1): 0.3
        }

        self.Y_C = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.4,
            (1, 1): 0.3
        }

        self.X_Y_C = {
            (0, 0, 0): 0.02,
            (0, 0, 1): 0.08,
            (0, 1, 0): 0.08,
            (0, 1, 1): 0.12,
            (1, 0, 0): 0.08,
            (1, 0, 1): 0.12,
            (1, 1, 0): 0.32,
            (1, 1, 1): 0.18,
        }

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for (x, y), p_xy in X_Y.items():
            p_x = X[x]
            p_y = Y[y]
            if not np.isclose(p_xy, p_x * p_y):
                return True
        return False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for (x, y, c), p_xyc in X_Y_C.items():
            p_xc = X_C[(x, c)]
            p_yc = Y_C[(y, c)]
            p_c = C[c]

            expected = (p_xc * p_yc) / p_c
            if not np.isclose(p_xyc, expected):
                return False
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    var = std ** 2
    denominator = np.sqrt(2 * np.pi * var)
    numerator = np.exp(- ((x - mean) ** 2) / (2 * var))
    p = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.features = self.dataset[:, :-1]
        self.features_means = np.mean(self.features, axis=0)
        self.features_stds = np.std(self.features, axis=0)
        self.num_class_samples = self.dataset.shape[0]
        self.total_samples = dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.num_class_samples / self.total_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = 1
        for i in range(len(x)):
            likelihood *= normal_pdf(x[i], self.features_means[i], self.features_stds[i])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = self.get_instance_likelihood(x)
        prior = self.get_prior()
        joint_prob = prior * likelihood
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        joint_0 = self.ccd0.get_instance_joint_prob(x)
        joint_1 = self.ccd1.get_instance_joint_prob(x)
        if joint_0 > joint_1:
            pred = 0
        else:
            pred = 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    d = len(x)
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)
    diff = x - mean

    norm_const = 1 / (np.sqrt(((2 * np.pi) ** d) * det_cov))
    exponent = -0.5 * diff.T @ inv_cov @ diff
    pdf = norm_const * np.exp(exponent)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset[dataset[:, -1] == class_value]
        self.features = self.dataset[:, :-1]
        self.class_value = class_value
        self.mean = np.mean(self.features, axis=0)
        self.cov = np.cov(self.features, rowvar=False)
        self.num_class_samples = self.dataset.shape[0]
        self.total_samples = dataset.shape[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.num_class_samples / self.total_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob



def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    num_correct = 0
    num_total = test_set.shape[0]

    for row in test_set:
        x = row[:-1]
        y_true = row[-1]
        y_pred = map_classifier.predict(x)
        if y_pred == y_true:
            num_correct += 1

    acc = num_correct / num_total
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.num_class_samples = self.dataset.shape[0]
        self.total_samples = dataset.shape[0]

        self.feature_conditional_probs = {}
        self.feature_value_counts = {}
        num_features = dataset.shape[1] - 1

        for t in range(num_features):
            col_values = dataset[:, t]
            unique_vals = set(col_values)
            self.feature_value_counts[t] = unique_vals

            for val in unique_vals:
                count = np.sum(self.dataset[:, t] == val)
                V_t = len(unique_vals)
                prob = (count + 1) / (self.num_class_samples + V_t)
                self.feature_conditional_probs[(t, val)] = prob
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = self.num_class_samples / self.total_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = 1
        for t, val in enumerate(x):
            key = (t, val)
            if key in self.feature_conditional_probs:
                likelihood *= self.feature_conditional_probs[key]
            else:
                V_t = len(self.feature_value_counts[t])
                prob = 1 / (self.num_class_samples + V_t)
                likelihood *= prob
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return joint_prob
