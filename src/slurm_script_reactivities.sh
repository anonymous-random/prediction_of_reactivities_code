#!/bin/bash

: '
This SLURM script is used to generate multiple jobs at a time for a given analysis setting.
For the machine learning analysis, we added compatibility for running the analysis on a supercomputer cluster using
a SLURM script. This SLURM script is used to generate multiple jobs at a time for a given analysis setting.
Key parameters here are BASE_MINUTES and BASE_CPUS. These are the base values for the time and the number
of CPUs a certain job has. These parameters are adjusted dynamically based on the type of analysis (e.g.,
ML analysis using single_items need more time than scale_means, because we have twice as many features,
rfr and svr need more time than Lasso, etc.)

Changeable variables/settings from the command line or from a SLURM script are
    - analysis ("main", "suppl")
    - suppl_type ( only if analysis==suppl, e.g., "sep_ftf_cmc")
    - suppl_var (only in certain suppl analyses, e.g., "ftf")
    - study ("ssc", "mse")
    - esm_sample ("coco_int", "emotions", "coco_ut")
    - feature_inclusion_strategy ("single_items", "scale_means", "feature_selection")
    - model ("linear_baseline_model", "lasso", "rfr", "svr")
    - soc_int_var ("social_interaction", "interaction_quantity", "interaction_closeness", interaction_depth")
    - setting: parallelization settings
    - setting: if ia_values are computed
'
# Possible parameters
ANALYSIS="suppl"  # main, suppl
SUPPL_TYPE="add_wb_change" # "weighting_by_rel" "sep_ftf_cmc" "add_wb_change" "sep_pa_na"
SUPPL_VAR="x" # pa, na; ftf, cmc; random_slopes, ols_slopes
STUDIES=("mse") # ssc, mse
ESM_SAMPLES=("coco_int") # "emotions" "coco_ut" "coco_int"
FEATURE_STRATEGIES=("scale_means")  # single_items, scale_means, feature_selection
MODELS=("svr")  # linear_baseline_model, lasso, rfr, svr
# SOC_INT_VARS=("social_interaction" "interaction_quantity" "interaction_depth" "interaction_closeness") # current$

CALC_IA_VALUES="true"
PARALLELIZE_REPS="false"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"

BASE_MINUTES=1
BASE_CPUS=4

# Base Directory for Results
BASE_DIR="/scratch/hpc-prf-mldpr/tests_022024/"

# Loop over all combinations
for study in "${STUDIES[@]}"; do
   for sample in "${ESM_SAMPLES[@]}"; do

      # Determine which SOC_INT_VARS to use based on sample
      case $sample in
          "coco_int")
              CURRENT_SOC_INT_VARS=("social_interaction" "interaction_quantity" "interaction_depth" "interaction_closeness")
              ESM_MULT=1
              ;;
          "coco_ut")
              CURRENT_SOC_INT_VARS=("social_interaction")
              # If sep_ftf_cmc ftf, we can add the variable interaction_quantity to coco_ut
              if [[ "${ANALYSIS[0]}" == "suppl" && "${SUPPL_TYPE[0]}" == "sep_ftf_cmc" && ( "${SUPPL_VAR[0]}" == "ftf" || "${SUPPL_VAR[0]}" == "ftf_pa" ) ]]; then
                  CURRENT_SOC_INT_VARS+=("interaction_quantity")
              fi
              ESM_MULT=1
              ;;
          "emotions")
              CURRENT_SOC_INT_VARS=("social_interaction" "interaction_quantity")
              ESM_MULT=1
              ;;
      esac

      # For the 'mse' study, we don't need to consider the SOC_INT_VARS
      if [ "$study" == "mse" ]; then
          CURRENT_SOC_INT_VARS=("")
      fi
      # Adjust the CPUs and time dynamically based on fis and model
      for strategy in "${FEATURE_STRATEGIES[@]}"; do
         STRATEGY_MULT=1

         case $strategy in
                "feature_selection") STRATEGY_MULT=10 ;;
         esac

         for model in "${MODELS[@]}"; do

         if [[ "$model" == "linear_baseline_model" && "$strategy" != "scale_means" ]]; then
             continue
         fi

            case $model in
                "lasso") MODEL_MULT=1 ;;
                "svr") MODEL_MULT=10 ;;
                "rfr") MODEL_MULT=5 ;;
                *) MODEL_MULT=1 ;;
            esac

            for soc_int_var in "${CURRENT_SOC_INT_VARS[@]}"; do

              TOTAL_MINUTES=$((BASE_MINUTES * ESM_MULT * STRATEGY_MULT * MODEL_MULT))
              TOTAL_CPUS=$((BASE_CPUS * ESM_MULT * MODEL_MULT))  # 40 CPUs is the limit, use strategy_mult only for time

              # Convert the total minutes to the HH:MM:SS format
              HOURS=$((TOTAL_MINUTES / 60))
              MINUTES=$((TOTAL_MINUTES % 60))
              TIMELIMIT=$(printf "%02d:%02d:00" $HOURS $MINUTES)

              # Create the necessary directory structure for results using the provided logic:
              if [ "$ANALYSIS" == "main" ]; then
                  if [ "$study" == "ssc" ]; then
                      RESULT_DIR="${BASE_DIR}/main_results/${study}/${sample}/${strategy}/${model}/${soc_int_var}"
                  else
                      RESULT_DIR="${BASE_DIR}/main_results/${study}/${sample}/${strategy}/${model}"
                  fi
                  elif [ "$ANALYSIS" == "suppl" ]; then
                      # Initially, set the path to BASE_DIR and SUPPL_TYPE
                      RESULT_DIR="${BASE_DIR}/${SUPPL_TYPE[0]}"

                      # Check and add SUPPL_VAR if it's set, not empty, and SUPPL_TYPE isn't 'add_wb_change'
                      if [ -n "${SUPPL_VAR[0]}" ] && [ "${SUPPL_TYPE[0]}" != "add_wb_change" ]; then
                          RESULT_DIR="${RESULT_DIR}/${SUPPL_VAR[0]}"
                      fi

                      # Add the remaining common elements
                      RESULT_DIR="${RESULT_DIR}/${study}/${sample}/${strategy}/${model}"

                      # Add soc_int_var for 'ssc' study, except when SUPPL_TYPE is 'add_wb_change'
                      if [ "$study" == "ssc" ] && [ "${SUPPL_TYPE[0]}" != "add_wb_change" ]; then
                          RESULT_DIR="${RESULT_DIR}/${soc_int_var}"
                      fi

                      # Check if SUPPL_TYPE is among the accepted values
                      if [[ ! "${SUPPL_TYPE[0]}" =~ ^(sep_ftf_cmc|sep_pa_na|weighting_by_rel|add_wb_change)$ ]]; then
                          echo "Supplementary analysis not implemented!"
                          exit 1
                      fi
              fi

              mkdir -p "$RESULT_DIR"

              LOG_DIR="logs_022024"
              LOG_BASE_PATH="$LOG_DIR"

              # Append subdirectory for ANALYSIS[0]
              if [ -n "${ANALYSIS[0]}" ]; then
                  LOG_BASE_PATH="${LOG_BASE_PATH}/${ANALYSIS[0]}"
              fi

              # Append subdirectory for SUPPL_TYPE[0] if ANALYSIS is suppl
              if [ "${ANALYSIS[0]}" == "suppl" ] && [ -n "${SUPPL_TYPE[0]}" ]; then
                  LOG_BASE_PATH="${LOG_BASE_PATH}/${SUPPL_TYPE[0]}"
              fi

              # Append subdirectory for SUPPL_VAR[0] if defined
              if [ "${ANALYSIS[0]}" == "suppl" ] && [ "${SUPPL_TYPE[0]}" != "add_wb_change" ] && [ -n "${SUPPL_VAR[0]}" ]; then
                  LOG_BASE_PATH="${LOG_BASE_PATH}/${SUPPL_VAR[0]}"
              fi

              # Append subdirectory for study
              LOG_BASE_PATH="${LOG_BASE_PATH}/${study}"

              # Ensure the full log directory path exists
              mkdir -p "${LOG_BASE_PATH}"

              # Construct base job and log names
              JOB_LOG_BASE="${sample}_${strategy}_${model}"

              CURRENT_TIME=$(date "+%Y%m%d-%H%M%S")
              JOB_LOG_NAME="${SLURM_JOB_ID}_${JOB_LOG_BASE}_${CURRENT_TIME}"

              # The full path for the log file
              FULL_LOG_PATH_LOG="${LOG_BASE_PATH}/${JOB_LOG_NAME}.log"
              FULL_LOG_PATH_ERR="${LOG_BASE_PATH}/${JOB_LOG_NAME}.err"

              # Create a temporary SLURM script
              cat > tmp_slurm_script.sh << EOF
#!/bin/bash

# SLURM Directives
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=$TOTAL_CPUS # Using the above computed CPU number for every Job
#SBATCH -p normal
#SBATCH -t $TIMELIMIT  # Using the above computed TIMELIMIT for every Job
#SBATCH --mail-type ALL
#SBATCH --mail-user aeback.oh@gmail.com
#SBATCH -J ${JOB_LOG_BASE}
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}

# LOAD MODULES HERE IF REQUIRED
module load python

# Your Python analysis script, with arguments
python main.py \
    --study "$study" \
    --esm_sample "$sample" \
    --feature_inclusion_strategy "$strategy" \
    --model "$model" \
    --social_interaction_variable "$soc_int_var" \
    --output "$RESULT_DIR/" \
    --analysis "$ANALYSIS" \
    --suppl_type "$SUPPL_TYPE" \
    --suppl_var "$SUPPL_VAR" \
    --calc_ia_values "$CALC_IA_VALUES" \
    --parallelize_reps "$PARALLELIZE_REPS" \
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \
    --parallelize_shap "$PARALLELIZE_SHAP" \

EOF

            # Submit the job and capture the job ID
            JOB_ID=$(sbatch tmp_slurm_script.sh | awk '{print $4}')

            # Remove the temporary script
            rm tmp_slurm_script.sh

            # Script for logging job details
            cat > log_job_details.sh << EOF
            #!/bin/bash

            # Wait for the main job to complete
            sleep 10

            # Fetch and log the job details using sacct
            sacct -j $JOB_ID --format=JobID,JobName,Partition,State,Elapsed,ReqMem,MaxRSS,AllocCPUS,TotalCPU,UserCPU,SystemCPU,NTasks,CPUTime,CPUTimeRAW,NodeList,NNodes -l > ${FULL_LOG_PATH_LOG}_job_$JOB_ID.log

            echo "Job $JOB_ID details logged to ${FULL_LOG_PATH_LOG}_job_$JOB_ID.log"
EOF

            # Submit the logging job as a dependent job
            sbatch --dependency=afterany:$JOB_ID log_job_details.sh

            # Remove the logging script
            rm log_job_details.sh

            done
         done
      done
   done
done