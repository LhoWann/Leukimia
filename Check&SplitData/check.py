import os

def check_data_distribution(dataset_path):
    total_all = 0
    
    for class_name in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, class_name)
        
        if os.path.isdir(class_dir):
            num_images = len(os.listdir(class_dir))
            print(f"Kelas {class_name}: {num_images} gambar")
            total_all += num_images
            
    print(f"\nTotal keseluruhan data: {total_all} gambar")

dataset_path = 'ALL_IDB Dataset'
check_data_distribution(dataset_path)