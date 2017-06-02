data {
    int<lower=1> N; // number of nodes
    int<lower=1> S; // number of unique stars
    int<lower=1> V; // maximum number of visits to any star
    int<lower=1,upper=V> visits[S]; // number of visits to each star;
    int<lower=1> L; // maximum number of line transitions
    int<lower=1,upper=L> reported_lines[S, V];

    int<lower=0> TM; // total number of missing estimates
    
    real lower_abundance_bound; // lower uniform bound on missing data
    real upper_abundance_bound; // upper uniform bound on missing data

    matrix[N*L, V] abundance_estimates[S]; // node-reported abundance
    int is_missing[S, N*L, V]; // if a node estimate is missing (>0 means missing)
    
    matrix[N*L, V] abundance_estimate_prior_mu[S];
    matrix[N*L, V] abundance_estimate_prior_sigma[S];

    int<lower=0> C; // number of calibrators
    real prior_mu[C]; // prior expected values
    real prior_sigma[C]; // prior uncertainties
    int prior_indices[C]; // indices to apply priors
}

transformed data {
    int NL; // 1-d length of covariance matrix
    int TSV; // total number of star visits

    NL = N * L; // 1-d length of covariance matrix
    TSV = sum(visits);
}

parameters {
    real abundance[S];
    //real<lower=lower_abundance_bound, upper=upper_abundance_bound> bias[NL];
    //real<lower=lower_abundance_bound, upper=upper_abundance_bound> missing_estimates[TM];

    
    vector<lower=0.01, upper=upper_abundance_bound-lower_abundance_bound>[NL] sigma; // uncertainties in line abundance for each node

    // Cholesky factor of a correlation matrix.
    cholesky_factor_corr[NL] L_corr;
}

transformed parameters {
    cov_matrix[NL] Sigma; // covariance matrix

    matrix[NL, V] full_rank_estimates[S]; // array containing known (data) and
                                         // unknown (parameter) estimate

    Sigma = diag_pre_multiply(sigma, L_corr) 
          * diag_pre_multiply(sigma, L_corr)';
    
}

model {

    // Any priors on particular stars?
    for (c in 1:C) {
        int index;
        index = prior_indices[c];
        abundance[index] ~ normal(prior_mu[c], prior_sigma[c]);
    }



    //L_corr ~ lkj_corr_cholesky(2);

    {
        int tsv;
        tsv = 1;

        for (s in 1:S) { // For each star
            for (v in 1:visits[s]) { // For each visit.

                for (nl in 1:NL) {
                    full_rank_estimates[s, nl, v] ~ normal(abundance_estimate_prior_mu[s, nl, v], abundance_estimate_prior_sigma[s, nl, v]);
                }
                
                full_rank_estimates[s, :, v] ~ multi_normal(
                    rep_vector(abundance[s], NL)',
                    Sigma);
                
                tsv = tsv + 1;
            }
        }
    }
}

generated quantities {

}