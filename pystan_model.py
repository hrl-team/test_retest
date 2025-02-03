    # Pystan model code
model_reduced = """
    functions {

      real log_prob_fun(real beta, real Q1, real Q2, int action){
        
        real Q_action;
        
        // Assign Q value of action taken and unchosen action (counterfactual) to be updated as "next" later on in the function
        if (action == 0) {
          Q_action = Q1;
        } else {
          Q_action = Q2;
        }
        
        real lp = beta * Q_action - log(exp(beta*Q1) + exp(beta*Q2));
        
        return lp;
      }
      

      // Update the q_learning function to return the Q_next values
      vector q_learning(real alpha_conf, real alpha_disc, real alpha_v, real beta, real Q1, real Q2, real V, int action, real reward, real reward_unchosen, int information) {
        real Q_next_1;
        real Q_next_2;
        real Q_next_action;
        real Q_next_counterfactual;
        real V_next;
        real rewardV;
        real log_prob;
        real Q_action;
        real Q_counterfactual;

        // Assign Q value of action taken and unchosen action (counterfactual) to be updated as "next" later on in the function
        if (action == 0) {
          Q_action = Q1;
          Q_counterfactual = Q2;
        } else {
          Q_action = Q2;
          Q_counterfactual = Q1;
        }

        // Update state value V according to complete or partial information
        if (information == 1) {
          rewardV = 0.5 * (reward + reward_unchosen);
        } else {
          rewardV = 0.5 * (reward + Q_counterfactual);
        }

        V_next = V + alpha_v * (rewardV - V);  // Update V with the average of chosen and unchosen outcomes

        real prediction_error_chosen = reward - V_next - Q_action;

        // Q update of action taken
        Q_next_action = Q_action + alpha_conf * prediction_error_chosen * (prediction_error_chosen > 0)
                        + alpha_disc * prediction_error_chosen * (prediction_error_chosen < 0);

        if (information == 1) {
          real prediction_error_unchosen = reward_unchosen - V_next - Q_counterfactual;

          // Q update of counterfactual action
          Q_next_counterfactual = Q_counterfactual + alpha_conf * prediction_error_unchosen * (prediction_error_unchosen < 0)
                                  + alpha_disc * prediction_error_unchosen * (prediction_error_unchosen > 0);
        } else {
        Q_next_counterfactual = Q_counterfactual;
        }

        // Re-assign Q values of action taken and unchosen action (counterfactual) after update
        if (action == 0) {
          Q_next_1 = Q_next_action;
          Q_next_2 = Q_next_counterfactual;
        } else {
          Q_next_2 = Q_next_action;
          Q_next_1 = Q_next_counterfactual;
        }

        return to_vector([Q_next_1, Q_next_2, V_next]);
      }
    }

    data {
      int<lower=1> num_experiments;
      int<lower=1> num_subjects;
      int<lower=1> num_rounds;
      int<lower=1> num_trials;
      array[num_subjects, num_experiments, num_rounds, num_trials] real<lower=-0.5, upper=0.5> Outcome;
      array[num_subjects, num_experiments, num_rounds, num_trials] real<lower=-0.5, upper=0.5> OutcomeUnchosen;  // Include OutcomeUnchosen
      array[num_subjects, num_experiments, num_rounds, num_trials] int<lower=0, upper=1> Information;
      array[num_subjects, num_experiments, num_rounds, num_trials] int<lower=0, upper=1> Action;
    }

    parameters {

      // Population-level parameters
      real<lower=0.01> a_population_alpha_conf;  // Adjust the lower limit as needed
      real<lower=0.01> a_population_alpha_disc;
      real<lower=0.01> a_population_alpha_v;
      real<lower=0.01> a_population_beta;

      real<lower=0.01> b_population_alpha_conf;  // Adjust the lower limit as needed
      real<lower=0.01> b_population_alpha_disc;
      real<lower=0.01> b_population_alpha_v;
      real<lower=0.01> b_population_beta;

      // Individual-level parameters
      array[num_subjects, num_experiments] real<lower=0, upper=1> alpha_conf;
      array[num_subjects, num_experiments] real<lower=0, upper=1> alpha_disc;
      array[num_subjects, num_experiments] real<lower=0, upper=1> alpha_v;
      array[num_subjects, num_experiments] real<lower=0> beta_par;

    }

    model {
      
      // Priors
      a_population_alpha_conf ~ gamma(2.1, 1);  // mode at 1.1
      a_population_alpha_disc ~ gamma(2.1, 1); //mode at 1.1
      a_population_alpha_v ~ gamma(2.1, 1);
      a_population_beta ~ gamma(2.2, 1); //mode at 1.2

      b_population_alpha_conf ~ gamma(2.1, 1);  // mode at 1.1
      b_population_alpha_disc ~ gamma(2.1, 1); //mode at 1.1
      b_population_alpha_v ~ gamma(2.1, 1);
      b_population_beta ~ gamma(6, 1); //mode at 5
      
      // to_vector(to_matrix(alpha_conf)) ~ beta(a_population_alpha_conf, b_population_alpha_conf);
      // to_vector(to_matrix(alpha_disc)) ~ beta(a_population_alpha_disc, b_population_alpha_disc);
      // to_vector(to_matrix(alpha_v)) ~ beta(a_population_alpha_v, b_population_alpha_v);
      // to_vector(to_matrix(beta_par)) ~ gamma(a_population_beta, b_population_beta);
    
      for (i in 1:num_subjects) {
        for (experiment in 1:num_experiments) {
          alpha_conf[i, experiment] ~ beta(a_population_alpha_conf, b_population_alpha_conf);
          alpha_disc[i, experiment] ~ beta(a_population_alpha_disc, b_population_alpha_disc);
          alpha_v[i, experiment] ~ beta(a_population_alpha_v, b_population_alpha_v);
          beta_par[i, experiment] ~ gamma(a_population_beta, b_population_beta);
        }
      }
      
      for (i in 1:num_subjects) {
      
        for (experiment in 1:num_experiments) {
          
          for (r in 1:num_rounds) {
        
            real Q_1;
            real Q_2;
            real V; // Added initialization for state value
        
            V = 0; // Initialize state value for each individual
            Q_1 = 0; // Initialize Q_1 for each individual
            Q_2 = 0; // Initialize Q_2 for each individual
        
            for (t in 1:num_trials) {

              target += log_prob_fun(beta_par[i, experiment], Q_1, Q_2, Action[i, experiment, r, t]); // Add log_prob to the target

              // Call q_learning and get the updated Q_next values        
              vector[3] outputs = q_learning(alpha_conf[i, experiment], alpha_disc[i, experiment], alpha_v[i, experiment], beta_par[i, experiment], Q_1, Q_2, V, Action[i, experiment, r, t], Outcome[i, experiment, r, t], OutcomeUnchosen[i, experiment, r, t], Information[i, experiment, r, t]);

              // Update Q_1 and Q_2 values with the new values
              Q_1 = outputs[1];
              Q_2 = outputs[2];
              V = outputs[3];
            }
          }
        }
      }
    }
    
    """

# Pystan model code
# Model based on Gamma distributions as priors

model_ind = """
    functions {

      real log_prob_fun(real beta, real Q1, real Q2, int action){
        
        real Q_action;
        
        // Assign Q value of action taken and unchosen action (counterfactual) to be updated as "next" later on in the function
        if (action == 0) {
          Q_action = Q1;
        } else {
          Q_action = Q2;
        }
        
        real lp = beta * Q_action - log(exp(beta*Q1) + exp(beta*Q2));
        
        return lp;
      }
      

      // Update the q_learning function to return the Q_next values
      vector q_learning(real alpha_conf, real alpha_disc, real alpha_v, real beta, real Q1, real Q2, real V, int action, real reward, real reward_unchosen, int information) {
        real Q_next_1;
        real Q_next_2;
        real Q_next_action;
        real Q_next_counterfactual;
        real V_next;
        real rewardV;
        real log_prob;
        real Q_action;
        real Q_counterfactual;

        // Assign Q value of action taken and unchosen action (counterfactual) to be updated as "next" later on in the function
        if (action == 0) {
          Q_action = Q1;
          Q_counterfactual = Q2;
        } else {
          Q_action = Q2;
          Q_counterfactual = Q1;
        }

        // Update state value V according to complete or partial information
        if (information == 1) {
          rewardV = 0.5 * (reward + reward_unchosen);
        } else {
          rewardV = 0.5 * (reward + Q_counterfactual);
        }

        V_next = V + alpha_v * (rewardV - V);  // Update V with the average of chosen and unchosen outcomes

        real prediction_error_chosen = reward - V_next - Q_action;

        // Q update of action taken
        Q_next_action = Q_action + alpha_conf * prediction_error_chosen * (prediction_error_chosen > 0)
                        + alpha_disc * prediction_error_chosen * (prediction_error_chosen < 0);

        if (information == 1) {
          real prediction_error_unchosen = reward_unchosen - V_next - Q_counterfactual;

          // Q update of counterfactual action
          Q_next_counterfactual = Q_counterfactual + alpha_conf * prediction_error_unchosen * (prediction_error_unchosen < 0)
                                  + alpha_disc * prediction_error_unchosen * (prediction_error_unchosen > 0);
        } else {
        Q_next_counterfactual = Q_counterfactual;
        }

        // Re-assign Q values of action taken and unchosen action (counterfactual) after update
        if (action == 0) {
          Q_next_1 = Q_next_action;
          Q_next_2 = Q_next_counterfactual;
        } else {
          Q_next_2 = Q_next_action;
          Q_next_1 = Q_next_counterfactual;
        }

        return to_vector([Q_next_1, Q_next_2, V_next]);
      }
    }

    data {
      int<lower=1> num_experiments;
      int<lower=1> num_subjects;
      int<lower=1> num_rounds;
      int<lower=1> num_trials;
      array[num_subjects, num_experiments, num_rounds, num_trials] real<lower=-0.5, upper=0.5> Outcome;
      array[num_subjects, num_experiments, num_rounds, num_trials] real<lower=-0.5, upper=0.5> OutcomeUnchosen;  // Include OutcomeUnchosen
      array[num_subjects, num_experiments, num_rounds, num_trials] int<lower=0, upper=1> Information;
      array[num_subjects, num_experiments, num_rounds, num_trials] int<lower=0, upper=1> Action;
    }

    parameters {

      // Population-level parameters
      real<lower=0.01> a_a_population_alpha_conf;  // Adjust the lower limit as needed
      real<lower=0.01> a_a_population_alpha_disc;
      real<lower=0.01> a_a_population_alpha_v;
      real<lower=0.01> a_a_population_beta;

      real<lower=0.01> b_a_population_alpha_conf;  // Adjust the lower limit as needed
      real<lower=0.01> b_a_population_alpha_disc;
      real<lower=0.01> b_a_population_alpha_v;
      real<lower=0.01> b_a_population_beta;

      real<lower=0.01> a_b_population_alpha_conf;  // Adjust the lower limit as needed
      real<lower=0.01> a_b_population_alpha_disc;
      real<lower=0.01> a_b_population_alpha_v;
      real<lower=0.01> a_b_population_beta;

      real<lower=0.01> b_b_population_alpha_conf;  // Adjust the lower limit as needed
      real<lower=0.01> b_b_population_alpha_disc;
      real<lower=0.01> b_b_population_alpha_v;
      real<lower=0.01> b_b_population_beta;

      // Individual-level hyperparameters
      array[num_subjects] real<lower=0.01> a_subject_alpha_conf;  // Adjust the lower limit as needed
      array[num_subjects] real<lower=0.01> a_subject_alpha_disc;
      array[num_subjects] real<lower=0.01> a_subject_alpha_v;
      array[num_subjects] real<lower=0.01> a_subject_beta;

      array[num_subjects] real<lower=0.01> b_subject_alpha_conf;  // Adjust the lower limit as needed
      array[num_subjects] real<lower=0.01> b_subject_alpha_disc;
      array[num_subjects] real<lower=0.01> b_subject_alpha_v;
      array[num_subjects] real<lower=0.01> b_subject_beta;

      // Individual-level parameters
      array[num_subjects, num_experiments] real<lower=0, upper=1> alpha_conf;
      array[num_subjects, num_experiments] real<lower=0, upper=1> alpha_disc;
      array[num_subjects, num_experiments] real<lower=0, upper=1> alpha_v;
      array[num_subjects, num_experiments] real<lower=0> beta_par;

    }

    model {
      
      // Priors
      
      // Priors population
      a_a_population_alpha_conf ~ gamma(3.1, 1);  // mode at 2.1
      a_a_population_alpha_disc ~ gamma(3.1, 1); //
      a_a_population_alpha_v ~ gamma(3.1, 1);
      a_a_population_beta ~ gamma(3.2, 1); //mode at 2.2

      b_a_population_alpha_conf ~ gamma(2, 1);  // mode at 1
      b_a_population_alpha_disc ~ gamma(2, 1); //
      b_a_population_alpha_v ~ gamma(2, 1);
      b_a_population_beta ~ gamma(2, 1); //mode at 1

      a_b_population_alpha_conf ~ gamma(3.1, 1);  // mode at 2.1
      a_b_population_alpha_disc ~ gamma(3.1, 1); //
      a_b_population_alpha_v ~ gamma(3.1, 1);
      a_b_population_beta ~ gamma(7, 1); //mode at 6
      
      b_b_population_alpha_conf ~ gamma(2, 1);  // mode at 1
      b_b_population_alpha_disc ~ gamma(2, 1); //
      b_b_population_alpha_v ~ gamma(2, 1);
      b_b_population_beta ~ gamma(2, 1); //    
      
      // Priors subject

      for (i in 1:num_subjects) {
      
        a_subject_alpha_conf[i] ~ gamma(a_a_population_alpha_conf, b_a_population_alpha_conf);  // 2.1, 1 -> mode at 1.1
        a_subject_alpha_disc[i] ~ gamma(a_a_population_alpha_disc, b_a_population_alpha_disc); //
        a_subject_alpha_v[i] ~ gamma(a_a_population_alpha_v, b_a_population_alpha_v);
        a_subject_beta[i] ~ gamma(a_a_population_beta, b_a_population_beta); // 2.2, 1 -> mode at 1.2

        b_subject_alpha_conf[i] ~ gamma(a_b_population_alpha_conf, b_b_population_alpha_conf);  // 2.1, 1 -> mode at 1.1
        b_subject_alpha_disc[i] ~ gamma(a_b_population_alpha_disc, b_b_population_alpha_disc); //
        b_subject_alpha_v[i] ~ gamma(a_b_population_alpha_v, b_b_population_alpha_v);
        b_subject_beta[i] ~ gamma(a_b_population_beta, b_b_population_beta); // 6, 1 -> mode at 5
      
      }
      
      
      for (i in 1:num_subjects) {
      
        for (experiment in 1:num_experiments) {

          alpha_conf[i, experiment] ~ beta(a_subject_alpha_conf, b_subject_alpha_conf); // 1.1, 1.1
          alpha_disc[i, experiment] ~ beta(a_subject_alpha_disc, b_subject_alpha_disc);
          alpha_v[i, experiment] ~ beta(a_subject_alpha_v, b_subject_alpha_v);
          beta_par[i, experiment] ~ gamma(a_subject_beta, b_subject_beta); // 1.2, 5
      
        }
      }
      
      for (i in 1:num_subjects) {
      
        for (experiment in 1:num_experiments) {
          
          for (r in 1:num_rounds) {
        
            real Q_1;
            real Q_2;
            real V; // Added initialization for state value
        
            V = 0; // Initialize state value for each individual
            Q_1 = 0; // Initialize Q_1 for each individual
            Q_2 = 0; // Initialize Q_2 for each individual
        
            for (t in 1:num_trials) {

              target += log_prob_fun(beta_par[i, experiment], Q_1, Q_2, Action[i, experiment, r, t]); // Add log_prob to the target

              // Call q_learning and get the updated Q_next values        
              vector[3] outputs = q_learning(alpha_conf[i, experiment], alpha_disc[i, experiment], alpha_v[i, experiment], beta_par[i, experiment], Q_1, Q_2, V, Action[i, experiment, r, t], Outcome[i, experiment, r, t], OutcomeUnchosen[i, experiment, r, t], Information[i, experiment, r, t]);

              // Update Q_1 and Q_2 values with the new values
              Q_1 = outputs[1];
              Q_2 = outputs[2];
              V = outputs[3];
            }
          }
        }
      }
    }
    
    """

# Pystan model code
# Model based on normal distributions as priors, as in Schurr et al. 2024

model_independent_normal = """
    functions {

      real partial_sum(array[,,,] int Action, int start, int end, array[,,,] int Information, array[,,,] real Outcome, array[,,,] real OutcomeUnchosen, int num_subjects, int num_experiments, int num_rounds, int num_trials, array[,] real alpha_conf, array[,] real alpha_disc, array[,] real alpha_v, array[,] real beta_par) {

        real lt = 0;  
      
        for (i in 1:num_subjects) {
        // for (n in 1:(end-start+1)) {

          // int i = start + (n-1);
        
        //for (i in start:end) {
        
          for (experiment in 1:num_experiments) {               
            
            for (r in 1:num_rounds) {
          
              real Q_1;
              real Q_2;
              real V; // Added initialization for state value
          
              V = 0; // Initialize state value for each individual
              Q_1 = 0; // Initialize Q_1 for each individual
              Q_2 = 0; // Initialize Q_2 for each individual
          
              for (t in 1:num_trials) {

                lt += log_prob_fun(beta_par[i, experiment], Q_1, Q_2, Action[i, experiment, r, t]); // Add log_prob to the target

                // Call q_learning and get the updated Q_next values        
                vector[3] outputs = q_learning(alpha_conf[i, experiment], alpha_disc[i, experiment], alpha_v[i, experiment], beta_par[i, experiment], Q_1, Q_2, V, Action[i, experiment, r, t], Outcome[i, experiment, r, t], OutcomeUnchosen[i, experiment, r, t], Information[i, experiment, r, t]);

                // Update Q_1 and Q_2 values with the new values
                Q_1 = outputs[1];
                Q_2 = outputs[2];
                V = outputs[3];
              }
            }
          }
        }
        return lt;

      }

      real log_prob_fun(real beta, real Q1, real Q2, int action){
        
        real Q_action;
        
        // Assign Q value of action taken and unchosen action (counterfactual) to be updated as "next" later on in the function
        if (action == 0) {
          Q_action = Q1;
        } else {
          Q_action = Q2;
        }
        
        real lp = beta * Q_action - log(exp(beta*Q1) + exp(beta*Q2));
        
        return lp;
      }
      

      // Update the q_learning function to return the Q_next values
      vector q_learning(real alpha_conf, real alpha_disc, real alpha_v, real beta, real Q1, real Q2, real V, int action, real reward, real reward_unchosen, int information) {
        real Q_next_1;
        real Q_next_2;
        real Q_next_action;
        real Q_next_counterfactual;
        real V_next;
        real rewardV;
        real log_prob;
        real Q_action;
        real Q_counterfactual;

        // Assign Q value of action taken and unchosen action (counterfactual) to be updated as "next" later on in the function
        if (action == 0) {
          Q_action = Q1;
          Q_counterfactual = Q2;
        } else {
          Q_action = Q2;
          Q_counterfactual = Q1;
        }

        // Update state value V according to complete or partial information
        if (information == 1) {
          rewardV = 0.5 * (reward + reward_unchosen);
        } else {
          rewardV = 0.5 * (reward + Q_counterfactual);
        }

        V_next = V + alpha_v * (rewardV - V);  // Update V with the average of chosen and unchosen outcomes

        real prediction_error_chosen = reward - V_next - Q_action;

        // Q update of action taken
        Q_next_action = Q_action + alpha_conf * prediction_error_chosen * (prediction_error_chosen > 0)
                        + alpha_disc * prediction_error_chosen * (prediction_error_chosen < 0);

        if (information == 1) {
          real prediction_error_unchosen = reward_unchosen - V_next - Q_counterfactual;

          // Q update of counterfactual action
          Q_next_counterfactual = Q_counterfactual + alpha_conf * prediction_error_unchosen * (prediction_error_unchosen < 0)
                                  + alpha_disc * prediction_error_unchosen * (prediction_error_unchosen > 0);
        } else {
        Q_next_counterfactual = Q_counterfactual;
        }

        // Re-assign Q values of action taken and unchosen action (counterfactual) after update
        if (action == 0) {
          Q_next_1 = Q_next_action;
          Q_next_2 = Q_next_counterfactual;
        } else {
          Q_next_2 = Q_next_action;
          Q_next_1 = Q_next_counterfactual;
        }

        return to_vector([Q_next_1, Q_next_2, V_next]);
      }
    }

    data {
      int<lower=1> num_experiments;
      int<lower=1> num_exp_samples;
      int<lower=1> num_subjects;
      int<lower=1> num_rounds;
      int<lower=1> num_trials;
      array[num_subjects, num_experiments, num_rounds, num_trials] real<lower=-0.5, upper=0.5> Outcome;
      array[num_subjects, num_experiments, num_rounds, num_trials] real<lower=-0.5, upper=0.5> OutcomeUnchosen;
      array[num_subjects, num_experiments, num_rounds, num_trials] int<lower=0, upper=1> Information;
      array[num_subjects, num_experiments, num_rounds, num_trials] int<lower=0, upper=1> Action;
      int<lower=1> P;
    }

    parameters {
        vector[P] mu_pr;                              // 4 parameters 
        vector<lower=0.0>[P] sigma_pr;                   // 4 parameters
        array[P, num_subjects] real mu_pr_sub;              // Subject-level means for each parameter
        array[P] real<lower=0.0> sigma_pr_r;                  // Subject-level standard deviations

        array[num_subjects, num_exp_samples] real alpha_conf_pr;   // Prior for alpha_conf
        array[num_subjects, num_exp_samples] real alpha_disc_pr;   // Prior for alpha_disc
        array[num_subjects, num_exp_samples] real alpha_v_pr;      // Prior for alpha_v
        array[num_subjects, num_exp_samples] real beta_par_pr;     // Prior for beta_par
    }

    transformed parameters {
    
        array[num_subjects, num_exp_samples] real<lower=0, upper=1> alpha_conf;
        array[num_subjects, num_exp_samples] real<lower=0, upper=1> alpha_disc;
        array[num_subjects, num_exp_samples] real<lower=0, upper=1> alpha_v;
        array[num_subjects, num_exp_samples] real<lower=0> beta_par;

      // Transform parameters using priors
      for (n in 1:num_subjects) {
        for (w in 1:num_exp_samples) {
          alpha_conf[n,w] = Phi(mu_pr_sub[1,n] + sigma_pr_r[1]*alpha_conf_pr[n,w]);
          alpha_disc[n,w] = Phi(mu_pr_sub[2,n] + sigma_pr_r[2]*alpha_disc_pr[n,w]);
          alpha_v[n,w] = Phi(mu_pr_sub[3,n] + sigma_pr_r[3]*alpha_v_pr[n,w]);
          beta_par[n,w] = exp(mu_pr_sub[4,n] + sigma_pr_r[4]*beta_par_pr[n,w]);
          }
        }
      }

    model {
        // Hyperparameters
        mu_pr[1] ~ normal(0, 1.0);
        mu_pr[2] ~ normal(0, 1.0);
        mu_pr[3] ~ normal(0, 1.0);
        mu_pr[4] ~ normal(0, 1.0);

        sigma_pr ~ normal(0, 1);

        // Subject-level parameters
        for (p in 1:P) {
            to_vector(mu_pr_sub[p,]) ~ normal(mu_pr[p], sigma_pr[p]);
        }

        sigma_pr_r ~ normal(0, 1);

        // Individual parameters with Matt trick
        to_vector(to_matrix(alpha_conf_pr)) ~ normal(0, 1);
        to_vector(to_matrix(alpha_disc_pr)) ~ normal(0, 1);
        to_vector(to_matrix(alpha_v_pr)) ~ normal(0, 1);
        to_vector(to_matrix(beta_par_pr)) ~ normal(0, 1);
      
          target += reduce_sum(partial_sum, Action, num_subjects, Information, Outcome, OutcomeUnchosen, num_subjects, num_experiments, num_rounds, num_trials, alpha_conf, alpha_disc, alpha_v, beta_par);
    }
    
    """
