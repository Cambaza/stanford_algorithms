import timeit
from certifi import where

from tenacity import retry_unless_exception_type

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
    for i in range(1, len(x)):
        key = x[i]
        j = i-1
        while j >= 0 and key < x[j] :
                x[j + 1] = x[j]
                j -= 1
        x[j + 1] = key
    
    return x



def bubble_sort(x):
    n = len(x)
    for i in range(n):
        for j in range(n-i-1):
            if x[j] > x[j+1]:
                x[j], x[j+1] = x[j+1], x[j]
    return x

def merge_function(a, b):
    c = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            c.append(a[i])
            i += 1

        else:
            c.append(b[j])
            j += 1
            
        
    while i < len(a):
        c.append(a[i])
        i += 1
    
    while j < len(b):
        c.append(b[j])
        j += 1

    return c
#print(merge_function([1, 3, 8],[2, 5, 11]))
def merge_sort(x):
    if len(x) > 1:
        mid = len(x) // 2
        left = x[:mid]
        right = x[mid:]

        # Recursive call on each half
        merge_sort(left)
        merge_sort(right)
        print(x)
        i = j = k = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
              # The value from the left half has been used
              x[k] = left[i]
              # Move the iterator forward
              i += 1
            else:
                x[k] = right[j]
                j += 1
            # Move to the next slot
            k += 1

        # For all the remaining values
        while i < len(left):
            x[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            x[k]=right[j]
            j += 1
            k += 1


a = [5, 1, 4, 2, 8, 3, 9, 6, 7]
#print('Unsorted list: ', a)
#starttime = timeit.default_timer()
#print('Sorted python method', sorted(a))
#print("Time of execution", timeit.default_timer() - starttime)


sortings = {'Selection Sort': selection_sort,
'Insertion Sort': insertion_sort,
'Buble Sort': bubble_sort,
'Merge Sort': merge_sort}


#starttime = timeit.default_timer()
#print('Selection Sort', selection_sort(a))
#print("Time of execution", timeit.default_timer() - starttime)


#starttime = timeit.default_timer()
#print('Insertion Sort', insertion_sort(a))
#print("Time of execution", timeit.default_timer() - starttime)

#starttime = timeit.default_timer()
print('Bubble Sort', bubble_sort(a))
#print("Time of execution", timeit.default_timer() - starttime)


#starttime = timeit.default_timer()
print('Merge Sort', a)
#print("Time of execution", timeit.default_timer() - starttime)