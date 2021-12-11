import get_input_runsets_crt_opts
import submit_job_files_given_opts
import os
def main():
    dict_file = "gnss_code_design/neural_networks/params_dictionary"
    opt_strs, params_dict = get_input_runsets_crt_opts.main(dict_file)
    print(opt_strs)
    submit_job_files_given_opts.main(opt_strs.split("\n"), params_dict)

if __name__ == "__main__":
    main()
