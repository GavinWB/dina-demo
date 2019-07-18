from __future__ import print_function, division, unicode_literals
import numpy as np
from psy import EmDina, MlDina
from psy.utils import r4beta

import codecs, json

def estimate_func():
    # Hyper param
    m = 100 # Number of students
    n = 42 # Number of questions
    # grade 4: n = 42, grade 8: n = 49
    K = 15 # Number of latent knowledges

    # Input matrices
    # q_mat = np.random.binomial(1, 0.5, (K, n)) # Q-matrix: latent knowledges x questions
    # Read from file
    q_mat = np.loadtxt('grade4-matrix.txt', dtype = int) # Q-matrix: latent knowledges x questions
    a_mat = np.random.binomial(1, 0.7, (m, K)) # A-matrix: student x latent knowledges

    # Guess and no slip (which is 1 - slip) probabilities for each question
    # Samples are drawn from beta distributions
    g = r4beta(1, 2, 0, 0.6, (1, n))
    no_s = r4beta(2, 1, 0.4, 1, (1, n))

    # Estimate the correct guessing and no_slipping probabilites based on each student test result
    # Using the EM algorithm (Estimation - Maximazation)
    temp = EmDina(attrs = q_mat)
    yita = temp.get_yita(a_mat)
    p_val = temp.get_p(yita, guess = g, no_slip = no_s)
    score = np.random.binomial(1, p_val)

    em_dina = EmDina(attrs = q_mat, score = score)
    est_no_s, est_g = em_dina.em()

    # Estimate student skill state
    dina_est = MlDina(guess = est_g, no_slip = est_no_s, attrs = q_mat, score = score)
    est_skills = dina_est.solve()

    print("done estimating")

    # return est_skills
    print(est_skills.shape)
    print(est_skills)

    # The Hamming distance between each student skill state and the question set will be calculated
    # and each student will receive the questions with largest Hamming distances

def output_func(data):
    file_path = "out.json"
    json.dump(data.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    print("done writing")

estimate_func()
