read -p "Write the number of codes (satellites): " K
read -p "Write the period of codes: " L
read -p "Write the objective function no: " obj_no
sbatch ./exec_search.sh $K $L $obj_no
echo "tail -n100 -f data/obj${obj_no}/sigm_bern_nes_model/${K}_${L}/init${init_mode}_sgs1.0_lr0.01_eps500000_smps${mc_samps}_run${run_name}/main_logs.txt"