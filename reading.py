    # ordered_file_list = []

    # filetype_str = '.I16'
    # start_image = 'Z0.I16'
    # for filename in os.listdir(path_):
    #     if start_image in filename:
    #         idx = filename.find(filetype_str)
    #         file_prefix = filename[:idx-1]
    #         i=0
    #         while True:
    #             check_filename = f'{file_prefix}{i}{filetype_str}'
    #             if os.path.exists(os.path.join(path_, check_filename)):
    #                 i+=1
    #                 #print(check_filename in ordered_file_list)
    #                 if check_filename not in ordered_file_list:
    #                     ordered_file_list.append(check_filename)
    #             else:
    #                 break
    # for filename in os.listdir(path_):
    #     if filetype_str in filename:
    #         i=0
    #         idx = filename.find(filetype_str)
    #         file_prefix = filename[:idx-1]
    #         while True:
    #             check_filename = f'{file_prefix}{i}{filetype_str}'
    #             if os.path.exists(os.path.join(path_, check_filename)):
    #                 i+=1
    #                 #print(check_filename in ordered_file_list)
    #                 if check_filename not in ordered_file_list:
    #                     ordered_file_list.append(check_filename)
    #             else:
    #                 break

    # print(ordered_file_list)
