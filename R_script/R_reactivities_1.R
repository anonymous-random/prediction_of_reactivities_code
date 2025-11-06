# ============================================================
# Multilevel CFA + Within-Person Reliability of Scale Scores (Mean of Items) for PA/NA
#
# Dependencies: lavaan (>= 0.6-17), semTools (optional), reticulate (for open pickle files)
# ============================================================
# install.packages(c("lavaan", "semTools", "reticulate"))
# reticulate::py_install("pandas")

# ============================================================
# Automated Multilevel CFA Reliability Analysis
#   - Iterates across all .pkl files in a directory
#   - Extracts corresponding PA/NA items for a given data subset 
#   - Runs within-person reliabilities for PA and NA
# ============================================================

suppressPackageStartupMessages({
  library(lavaan)
  if (requireNamespace("semTools", quietly = TRUE)) invisible(TRUE)
  if (requireNamespace("reticulate", quietly = TRUE)) invisible(TRUE)
})

# ------------------------------------------------------------
# Load .pkl -> data.frame
# ------------------------------------------------------------
load_pkl_dataframe <- function(path) {
  if (!requireNamespace("reticulate", quietly = TRUE))
    stop("Install 'reticulate' to load .pkl files.")
  if (!file.exists(path))
    stop("File not found: ", path)
  reticulate::py_config()
  pd <- reticulate::import("pandas", delay_load = TRUE)
  as.data.frame(pd$read_pickle(path))
}

# ------------------------------------------------------------
# Multilevel CFA model syntax
# ------------------------------------------------------------
mlcfa_syntax <- function(items, latent = "Affect") {
  ind <- paste(items, collapse = " + ")
  paste0(
    "level: 1\n  ", latent, " =~ ", ind, "\n\n",
    "level: 2\n  ", latent, " =~ ", ind, "\n"
  )
}

# ------------------------------------------------------------
# Fit two-level CFA
# ------------------------------------------------------------
fit_mlcfa <- function(data, id_var, items, latent = "Affect",
                      estimator = "MLR") {
  model <- mlcfa_syntax(items, latent)
  lavaan::sem(model = model,
              data = data,
              cluster = id_var,
              estimator = estimator,
              missing = "fiml",
              fixed.x = FALSE,
              std.lv = TRUE)
}

# ------------------------------------------------------------
# Extract model matrices for one level
# ------------------------------------------------------------
extract_level_mats <- function(fit, items, latent, level = c("within","between")) {
  level <- match.arg(level)
  est <- lavInspect(fit, "est")
  lev_idx <- if (level == "within") 1 else 2
  Lambda <- est[[lev_idx]]$lambda[items, latent, drop = FALSE]
  Theta  <- est[[lev_idx]]$theta[items, items, drop = FALSE]
  Phi    <- est[[lev_idx]]$psi[latent, latent, drop = FALSE]
  list(Lambda = Lambda, Theta = Theta, Phi = Phi)
}

# ------------------------------------------------------------
# Composite reliability for unit-weighted scores
# ------------------------------------------------------------
composite_reliability_unit <- function(Lambda, Theta, Phi, w) {
  w <- matrix(w, ncol = 1)
  VT <- t(w) %*% Lambda %*% Phi %*% t(Lambda) %*% w
  VO <- t(w) %*% (Lambda %*% Phi %*% t(Lambda) + Theta) %*% w
  as.numeric(VT / VO)
}

# ------------------------------------------------------------
# Try semTools::compRelSEM(); record if used
# ------------------------------------------------------------
compRel_via_semTools_or_fallback <- function(Lambda, Theta, Phi, w, fit = NULL, items = NULL, latent = NULL) {
  used_semTools <- FALSE
  reliability <- NA_real_
  
  # Ensure names for consistency
  if (!is.null(items)) {
    rownames(Lambda) <- items
    colnames(Theta) <- rownames(Theta) <- items
    names(w) <- items
  }
  
  # --- Attempt 1: semTools using Lambda, Theta, W ---
  if (requireNamespace("semTools", quietly = TRUE)) {
    out_try <- try({
      semTools::compRelSEM(Lambda = Lambda, Theta = Theta, W = matrix(w, ncol = 1))
    }, silent = TRUE)
    if (!inherits(out_try, "try-error")) {
      used_semTools <- TRUE
      reliability <- as.numeric(out_try)[1]
      return(list(reliability = reliability, used_semTools = used_semTools))
    }
    
    # --- Attempt 2: semTools using lavaan object (safer API) ---
    if (!is.null(fit) && !is.null(latent)) {
      out_try2 <- try({
        semTools::compRelSEM(fit, latent = latent, level = 1)
      }, silent = TRUE)
      if (!inherits(out_try2, "try-error")) {
        used_semTools <- TRUE
        reliability <- as.numeric(out_try2)[1]
        return(list(reliability = reliability, used_semTools = used_semTools))
      }
    }
  }
  
  # --- Fallback analytic computation ---
  reliability <- composite_reliability_unit(Lambda, Theta, Phi, w)
  list(reliability = reliability, used_semTools = used_semTools)
}

# ------------------------------------------------------------
# Fit model + compute within-person reliability
# ------------------------------------------------------------
run_mlcfa_scale_reliability <- function(data, id_var, items, latent,
                                        estimator = "MLR", weights = NULL) {
  if (is.null(weights)) weights <- rep(1 / length(items), length(items))
  fit <- fit_mlcfa(data, id_var, items, latent, estimator)
  mats_w <- extract_level_mats(fit, items, latent, "within")
  
  res <- compRel_via_semTools_or_fallback(
    Lambda = mats_w$Lambda,
    Theta = mats_w$Theta,
    Phi = mats_w$Phi,
    w = weights,
    fit = fit,
    items = items,
    latent = latent
  )
  res
}

# ------------------------------------------------------------
# Main function: iterate across all .pkl files and write table
# ------------------------------------------------------------
run_all_reliabilities <- function(dir_path, id_var = "id", sample_n = 1000,
                                  out_file = "reliability_summary.csv") {
  files <- list.files(dir_path, pattern = "\\.pkl$", full.names = TRUE)
  if (length(files) == 0) stop("No .pkl files found in: ", dir_path)
  
  results_tbl <- data.frame(
    file = character(),
    social_variable = character(),
    reliability_PA = numeric(),
    reliability_NA = numeric(),
    used_semTools_PA = logical(),
    used_semTools_NA = logical(),
    n_rows = integer(),
    n_ids = integer(),
    stringsAsFactors = FALSE
  )
  
  for (file_path in files) {
    message("\nProcessing file: ", basename(file_path))
    df <- load_pkl_dataframe(file_path)
    
    # --- Infer social variable name ------------------------------------------
    filename <- basename(file_path)
    social_var <- sub("_preprocessed\\.pkl$", "", filename)
    social_var <- sub(".*interaction_", "interaction_", social_var)
    
    if (!grepl("^interaction_", social_var) && grepl("social_interaction", filename)) {
      social_var <- "social_interaction"
    }
    
    matched_col <- grep(social_var, names(df), value = TRUE)
    if (length(matched_col) == 0) {
      message("   No column matches inferred social variable: ", social_var)
      next
    }
    
    # --- Select relevant columns ---------------------------------------------
    pa_items <- grep("^state_pa", names(df), value = TRUE)
    na_items <- grep("^state_na", names(df), value = TRUE)
    keep_cols <- unique(c(id_var, matched_col, pa_items, na_items))
    sub_df <- df[keep_cols]
    sub_df <- sub_df[complete.cases(sub_df), ]
    
    # --- Optional random sample ----------------------------------------------
    if (!is.null(sample_n) && nrow(sub_df) > sample_n) {
      set.seed(123)
      sub_df <- sub_df[sample(nrow(sub_df), sample_n), ]
      message("  Sampled ", sample_n, " rows.")
    }
    
    # --- Keep only IDs with ≥2 observations ---------------------------------
    id_counts <- table(sub_df[[id_var]])
    valid_ids <- names(id_counts[id_counts > 1])
    sub_df <- subset(sub_df, sub_df[[id_var]] %in% valid_ids)
    
    if (nrow(sub_df) < 10) {
      message(" Not enough repeated observations for multilevel model.")
      next
    }
    
    # --- Run reliability analyses --------------------------------------------
    res_pa <- run_mlcfa_scale_reliability(sub_df, id_var, pa_items, "PosAffect")
    res_na <- run_mlcfa_scale_reliability(sub_df, id_var, na_items, "NegAffect")
    
    rel_pa <- res_pa$reliability
    rel_na <- res_na$reliability
    used_sem_pa <- res_pa$used_semTools
    used_sem_na <- res_na$used_semTools
    
    # --- Add row to summary table --------------------------------------------
    results_tbl <- rbind(
      results_tbl,
      data.frame(
        file = filename,
        social_variable = matched_col,
        reliability_PA = rel_pa,
        reliability_NA = rel_na,
        used_semTools_PA = used_sem_pa,
        used_semTools_NA = used_sem_na,
        n_rows = nrow(sub_df),
        n_ids = length(unique(sub_df[[id_var]])),
        stringsAsFactors = FALSE
      )
    )
    
    # --- Print summary -------------------------------------------------------
    cat("\n---------------------------------------------\n")
    cat("File: ", filename, "\n")
    cat("Social variable: ", matched_col, "\n")
    cat("Reliability (PA): ", sprintf("%.3f", rel_pa),
        " | semTools used: ", used_sem_pa, "\n")
    cat("Reliability (NA): ", sprintf("%.3f", rel_na),
        " | semTools used: ", used_sem_na, "\n")
    cat("---------------------------------------------\n")
  }
  
  # --- Write summary CSV -----------------------------------------------------
  out_path <- file.path(dir_path, out_file)
  write.csv(results_tbl, out_path, row.names = FALSE)
  message("\n Results written to: ", out_path)
  
  return(results_tbl)
}

# set wd to the dir where the states are stored, e.g., "../data/preprocessed/main/ssc/states"
results <- run_all_reliabilities(getwd(), id_var = "id", sample_n = 999999)  # basically no sampling
View(results)