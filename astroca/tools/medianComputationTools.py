from numba import njit, prange
import numpy as np



@njit
def quickselect_median(arr, n):
    """
    @brief Optimized median calculation for small arrays

    @details Uses quickselect to find median element without full sorting.
    Falls back to insertion sort for very small arrays (n ≤ 20).

    @param arr Input array (will be modified)
    @param n Number of elements to consider in array

    @return Median value
    """
    if n == 0:
        return 0
    if n == 1:
        return arr[0]

    # For small arrays, insertion sort is faster
    if n <= 20:
        # Insertion sort
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

        if n % 2 == 1:
            return arr[n // 2]
        else:
            return (arr[n // 2 - 1] + arr[n // 2]) / 2.0

    # For larger arrays, use quickselect
    # But in practice, our spherical neighborhoods are small
    return quickselect_kth(arr, n, n // 2)


@njit
def quickselect_kth(arr, n, k):
    """
    @brief Finds the k-th smallest element (0-indexed)

    @param arr Input array (will be modified)
    @param n Number of elements to consider
    @param k Index of desired element (0 = smallest)

    @return The k-th smallest element
    """
    left = 0
    right = n - 1

    while left < right:
        pivot_idx = partition(arr, left, right)
        if pivot_idx == k:
            return arr[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1

    return arr[k]


@njit
def partition(arr, left, right):
    """
    @brief Partition function for quickselect

    @param arr Array to partition
    @param left Left index
    @param right Right index (pivot position)

    @return Final pivot position
    """
    pivot = arr[right]
    i = left - 1

    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1