output_folder = 'a'
user_input = input(f"NOTE: The output folder {output_folder} exists, delete (d), resume (r) or quit (q): ")

if user_input == 'd':
    print('delete')
elif user_input == 'r':
    print('resume')