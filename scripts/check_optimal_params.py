import glob

vehicles = ['vehicle.volkswagen.t2',
        'vehicle.bmw.isetta',
        'vehicle.carlamotors.carlacola',
        'vehicle.jeep.wrangler_rubicon',
        'vehicle.nissan.patrol',
        'vehicle.tesla.cybertruck',
        'vehicle.yamaha.yzf',
        'vehicle.bh.crossbike']


if __name__ == "__main__":
    for vehicle in vehicles:
        flag = vehicle.replace('.', '_')
        files = glob.glob(flag + '*.txt')
        best_max_error = 10000
        best_mean_error = 10000
        best_max_error_file = ''
        best_mean_error_file = ''
        for file_name in files:
            with open(file_name, 'r') as file:
                lines = file.readlines()
                max_error = float(lines[0])
                mean_error = float(lines[1])

                if max_error < best_max_error:
                    best_max_error = max_error
                    best_max_error_file = file_name

                if mean_error < best_mean_error:
                    best_mean_error = mean_error
                    best_mean_error_file = file_name

        best_mean_error_file = best_mean_error_file[-15:-4].replace('_',' ').split()
        best_max_error_file = best_max_error_file[-15:-4].replace('_',' ').split()

        # print('{} best_max_error: {} value {}'.format(vehicle, best_max_error_file, best_max_error))
        print("model = '{}'\nself.KP = {}\nself.KI = {}\nself.KD = {}\nmean error {}".format(vehicle, best_mean_error_file[0], best_mean_error_file[1], best_mean_error_file[2], best_mean_error))
        # print("model = '{}'\nKP = {}\nKI = {}\nKD = {}".format(vehicle, best_mean_error_file[0], best_mean_error_file[1], best_mean_error_file[2]))
