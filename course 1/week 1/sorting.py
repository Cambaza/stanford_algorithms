import timeit

def selection_sort(x):
    # O(n^2)
    n = len(x)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if x[min_index] > x[j]:
                x[min_index], x[j] = x[j], x[min_index]
    return x


def insertion_sort(x):
    pass 


def bubble_sort(x):
    pass 

def merge_sort(x):
    pass




a = [5, 1, 4, 2, 8, 3, 9, 6, 7]
print('Unsorted list: ', a)
starttime = timeit.default_timer()
print('Sorted python method', sorted(a))
print("Time of execution", timeit.default_timer() - starttime)


sortings = {'Selection Sort': selection_sort,
'Insertion Sort': insertion_sort,
'Buble Sort': bubble_sort,
'Merge Sort': merge_sort}


starttime = timeit.default_timer()
print('Selection Sort', selection_sort(a))
print("Time of execution", timeit.default_timer() - starttime)