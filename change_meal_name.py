'''
將data中餐飲的詞曲代為'甲'
'''


def replace_meal_all():
    with open('newData_For_AskAns.txt', 'r') as f:
        f_meals = open('meals.txt', 'r')
        f_out = open('replaced_data.txt', 'w')
        line = f.readline()
        meals = f_meals.readlines()
        while line:
            if line.find("雲端咖啡館") == -1:
                for meal in iter(meals):
                    meal = meal.strip('\n')
                    num = line.find(meal)
                    while num != -1:
                        line = line[:num] + 'allkindofmeal' + line[num + len(meal):]
                        num = line.find(meal)

            f_out.write(line)
            line = f.readline()


def replace_meal_line(line, meals):
    order = []
    if line.find("雲端咖啡館") == -1:
        for i, _ in enumerate(meals):
            meals[i] = meals[i].strip('\n')
            num = line.find(meals[i])
            while num != -1:
                line = line[:num] + 'allkindofmeal' + line[num + len(meals[i]):]
                num = line.find(meals[i])
                order.append(meals[i])
    return order, line


def main():
    replace_meal_all()

if __name__ == '__main__':
    main()
