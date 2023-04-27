import pickle

def display_object_content(obj):
    attributes = [
        'all_gather_cost_dict',
        'all_reduce_cost_dict',
        'all_to_all_cost_dict',
        'reduce_scatter_cost_dict',
        'available_memory_per_device',
        'dot_cost_dict',
        'conv_cost_dict',
        'op_cost_dict',
    ]
    
    for attr in attributes:
        value = getattr(obj, attr)
        print(f"{attr}:")
        if isinstance(value, dict):
            for key, val in value.items():
                print(f"  {key}: {val}")
        else:
            print(f"  {value}")
        print()

with open('prof_database.pkl', 'rb') as f:
    prof_database = pickle.load(f)

print("Content of prof_database.pkl file:\n")
for key in prof_database.keys():
    print(f"Key: {key}")
    obj = prof_database[key]
    display_object_content(obj)
    print("-" * 50)
