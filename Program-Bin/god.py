import time

# The flag starts as False
exit_flag = False

def set_exit_flag():
    """This function exists but is never called in the program"""
    global exit_flag
    exit_flag = True
    print("Flag has been set by external means")

def infinite_prayer():
    arabic_prayer = "اللهم بارك لي فإني فقير محتاج إلى وصي"
    ending_praise = "سبحان الله وبحمده"
    
    count = 0
    while True:
        # Check if the flag has been mysteriously set
        if exit_flag:
            print(f"\n{ending_praise}")
            print("Program ended by divine intervention")
            return
            
        print(f"{arabic_prayer} - {count}")
        count += 1
        time.sleep(3)

# Run the program
if __name__ == "__main__":
    print("Starting infinite prayer...")
    print("Only external intervention can stop this program")
    infinite_prayer()