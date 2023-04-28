import numpy as np
'''
def stage_profile_result_str(stage_profile_result):
    total_initial_var_size = sum(stage_profile_result.initial_var_sizes)
    available_memory_str = (
        f"{stage_profile_result.available_memory / GB:.3f} GB"
        if stage_profile_result.available_memory is not None
        else "None"
    )
    return (
        f"StageProfileResult("
        f"available_memory={available_memory_str}, "
        f"initial_var_size={total_initial_var_size / GB:.3f} GB, "
        f"module_profile_results={stage_profile_result.module_profile_results})"
    )
'''

'''
def module_profile_result_str(module_profile_result):
    if module_profile_result is None or module_profile_result.invar_sizes is None:
        return "None"

    invar_size = sum(module_profile_result.invar_sizes)
    outvar_size = sum(module_profile_result.outvar_sizes)
    return (
        f"ModuleProfileResult("
        f"compute_cost={module_profile_result.compute_cost:.3f}, "
        f"peak_memory={module_profile_result.peak_memory / GB:.3f} GB, "
        f"invar_size={invar_size / GB:.3f} GB, "
        f"outvar_size={outvar_size / GB:.3f} GB, "
        f"temp_buffer_size={module_profile_result.temp_buffer_size / GB:.3f} GB, "
        f"available_memory={module_profile_result.available_memory / GB:.3f} GB)"
    )

'''

def module_profile_result_str(module_profile_result):
    if module_profile_result is None:
        return "None"

    invar_size = sum(module_profile_result.invar_sizes) if module_profile_result.invar_sizes is not None else None
    outvar_size = sum(module_profile_result.outvar_sizes) if module_profile_result.outvar_sizes is not None else None
    return (
        f"ModuleProfileResult("
        f"compute_cost={module_profile_result.compute_cost:.3f}, "
        f"peak_memory={module_profile_result.peak_memory / GB:.3f} GB, "
        f"temp_buffer_size={module_profile_result.temp_buffer_size / GB:.3f} GB, "
        f"invar_names={module_profile_result.invar_names}, "
        f"outvar_names={module_profile_result.outvar_names}, "
        f"invar_sizes={module_profile_result.invar_sizes}, "
        f"outvar_sizes={module_profile_result.outvar_sizes}, "
        f"donated_invars={module_profile_result.donated_invars}, "
        f"acc_grad_invars_indices={module_profile_result.acc_grad_invars_indices}, "
        f"acc_grad_outvars_indices={module_profile_result.acc_grad_outvars_indices}, "
        f"available_memory={module_profile_result.available_memory / GB:.3f} GB)"
    )



def stage_profile_result_str(stage_profile_result):
    total_initial_var_size = sum(stage_profile_result.initial_var_sizes)
    available_memory_str = (
        f"{stage_profile_result.available_memory / GB:.3f} GB"
        if stage_profile_result.available_memory is not None
        else "None"
    )
    module_profile_results_str = [
        module_profile_result_str(module_result)
        for module_result in stage_profile_result.module_profile_results
    ]
    return (
        f"StageProfileResult("
        f"available_memory={available_memory_str}, "
        f"initial_var_size={total_initial_var_size / GB:.3f} GB, "
        f"module_profile_results={module_profile_results_str})"
    )



def print_stage_profile_results(stage_profile_results):
    for key, stage_profile_result in stage_profile_results.items():
        print(f"{key}: {stage_profile_result_str(stage_profile_result)}")


name_list=["profile-results-2023-04-25-16-59-36.npy",
"profile-results-2023-04-25-16-15-47.npy",  "profile-results-2023-04-25-17-25-46.npy",
"profile-results-2023-04-25-16-38-00.npy",  "profile-results-2023-04-25-18-01-12.npy"]
path="./after-run-benchmark-grid-search-result/"
GB = 1024**3  # Add this line to define the GB constant
for filename in name_list:
    # Load the .npy file
    print(filename)
    data = np.load(path+filename,allow_pickle=True).item()
    # Loop through the dictionary keys and print the attributes of the corresponding StageProfileResult object
     # Loop through the dictionary keys and print the attributes of the corresponding StageProfileResult object
    print_stage_profile_results(data)
    '''
    for key, stage_profile_result in data.items():
        print(f"Key: {key}")
        #print(stage_profile_result)
        #print(stage_profile_result.__str__())  # Call the __str__ method of the StageProfileResult object
        #print_stage_profile_results(stage_profile_result)

        print(stage_profile_result_str(stage_profile_result))
        print()
    '''
    print(data)