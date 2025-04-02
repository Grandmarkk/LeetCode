import numpy as np
import math
from typing import List, Optional

# Single linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Binary tree node
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right

# Multi-children Tree Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

# 225
class MyStack:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)

    def pop(self) -> int:
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def empty(self) -> bool:
        return len(self.stack) == 0
    
# 303
class NumArray:

    def __init__(self, nums):
        self.sums = []
        curSum = 0
        for num in nums:
            self.sums.append(curSum + num)
            curSum += num

    def sumRange(self, left, right):
        if left != 0:
            return self.sums[right] - self.sums[left-1]
        else:
            return  self.sums[right]

# 981
class TimeMap:

    def __init__(self):
        self.content = {}
        
    def set(self, key: str, value: str, timestamp: int) -> None:
        if key in self.content:
            self.content[key].append((timestamp, value))
        else:
            self.content[key] = [(timestamp, value)]
        self.content[key].sort()
    def get(self, key: str, timestamp: int) -> str:
        if key in self.content:
            left = 0
            right = len(self.content[key]) - 1
            while left <= right:
                mid = (left + right) // 2
                cur = self.content[key][mid]
                if cur[0] > timestamp:
                    right = mid - 1
                elif cur[0] < timestamp:
                    left = mid + 1
                else:
                    return cur[1]
            return self.content[key][right][1] if self.content[key][right][0] <= timestamp else ""
        return ""


class Solution:
    # 1
    def twoSum(self, nums, target):
        '''
        Function Description:
            Find the two numbers that sum to the target number
        
        Input:
            nums: the list of numbers to find the answers in
            target: the target sum
        
        Output:
            a list of two indexes of the two numbers that would sum to target
        '''
        numDict = {}
        i = 0
        while i < len(nums):
            curRemainder = target - nums[i]
            if curRemainder in numDict:
                return [i, numDict[curRemainder]]
            numDict[nums[i]] = i
            i += 1
        return []
        
    # 2
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        curSum = (l1.val + l2.val) % 10
        carry = (l1.val + l2.val) // 10
        head = ListNode(curSum)
        temp = head
        l1 = l1.next
        l2 = l2.next
        while l1 and l2:
            curSum = (carry + l1.val + l2.val) % 10
            carry = (carry + l1.val + l2.val) // 10
            newNode = ListNode(curSum)
            head.next = newNode
            head = head.next
            l1 = l1.next
            l2 = l2.next
        while l1:
            curSum = (carry + l1.val) % 10
            carry = (carry + l1.val) // 10
            newNode =ListNode(curSum)
            head.next = newNode
            head = head.next
            l1 = l1.next
        while l2:
            curSum = (carry + l2.val) % 10
            carry = (carry + l2.val) // 10
            newNode =ListNode(curSum)
            head.next = newNode
            head = head.next
            l2 = l2.next
        if carry > 0:
            newNode = ListNode(1)
            head.next = newNode
        return temp

    # 9
    def isPalindrome(self, x):
        '''
        Function Description:
            determine if the number is a palindrome

        Input:
            x: the number that will be determined if is palindrome
        
        Output:
            boolean, true if is palindrome, false if not palindrome
        '''
        # negative number cannot be palindrome for the -
        if x < 0:
            return False
        x = str(x)
        return x == x[::-1]
    
    # 13
    def romanToInt(self, s):
        '''
        Function Description:
            convert a string representing a roman number to int

        Input:
            s: a string representing a roman number

        Output:
            integer form of the input roman number
        '''
        romanNumDict = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        ans = 0
        length = len(s)
        i = 0
        while i < length:
            # special cases
            if i != length - 1 and (romanNumDict[s[i]] < romanNumDict[s[i+1]]):
                ans += romanNumDict[s[i+1]] - romanNumDict[s[i]]
                i += 1
            else:
                ans += romanNumDict[s[i]]
            i += 1
        return ans
    
    # 14
    def longestCommonPrefix(self, strs):
        ans = []
        # get shortest word length
        minLen = len(strs[0])
        for word in strs:
            if len(word) < minLen:
                minLen = len(word)
        # scan
        i = 0
        while i < minLen:
            char = strs[0][i]
            j = 0
            while j < len(strs):
                if char != strs[j][i]:
                    return ''.join(ans)
                j += 1
            ans.append(char)
            i += 1
        return ''.join(ans)
    
    # 15
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        length = len(nums)
        for index, num in enumerate(nums):
            if num > 0:
                break
            if index > 0 and num == nums[index - 1]:
                continue
            left = index + 1
            right = length - 1
            # find two sum
            while left < right:
                curSum = nums[left] + nums[right] + num
                if curSum > 0:
                    right -= 1
                elif curSum < 0:
                    left += 1
                else:
                    ans.append([num, nums[left], nums[right]])
                    left += 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
        return ans
    
    # 19 
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        def getLength(head):
            if not head.next:
                return 1
            return 1 + getLength(head.next)
        
        def search(head, n, count):
            if not head:
                return
            if count == n:
                if head.next:
                    head.next = head.next.next
                else:
                    head.next =None
            search(head.next, n, count + 1)

        length = getLength(head)
        n = length - n
        # remove head
        if n == 0:
            return head.next
        temp = head
        search(temp, n, 0)
        return head

    # 20
    def isValid(self, s):
        '''
        Function Description:
            determine if the brackets are valid

        Input:
            s: a string of brackets

        Output:
            bool
        '''
        stack = []
        brackets = {')': '(', ']': '[', '}': '{'}
        opens = brackets.values()
        for bracket in s:
            if bracket in opens:
                stack.append(bracket)
            else:
                if len(stack) == 0:
                    return False
                if stack[-1] == brackets[bracket]:
                    stack.pop()
                else:
                    stack.append(bracket)
        return len(stack) == 0
    
    # 21
    def mergeTwoLists(self, list1, list2):
        '''
        Function Description:
            merge 2 sorted linked lists

        Input:
            2 linked lists

        Output:
            sorted linked list from the input lists
        '''
        mergedList = ListNode()
        temp = mergedList
        # compare 
        while list1 and list2:
            if list1.val < list2.val:
                mergedList.next = list1
                list1 = list1.next
            else:
                mergedList.next = list2
                list2 = list2.next
            mergedList = mergedList.next
        # 1 of the lists is empty
        while list1:
            mergedList.next = list1
            list1 = list1.next
            mergedList = mergedList.next
        while list2:
            mergedList.next = list2
            list2 = list2.next
            mergedList = mergedList.next
        return temp.next
    
    # 26
    def removeDuplicates(self, nums):
        '''
        Function Description:
            remove duplicated nums inplace

        Input:
            a list of ints

        Output:
            new length of the list
        '''
        fast = 1
        slow = 1
        numLen = len(nums)
        if numLen < 2:
            return numLen
        while fast < numLen:
            if nums[fast] != nums[fast-1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
    
    # 27
    def removeElement(self, nums, val):
        '''
        Function Description:
            remove element that equals to val in the array nums inplace

        Input:
            nums: a list of ints
            val: the val that's to be removed

        Output:
            new length of the array
        '''
        fast = 0
        slow = 0
        numLen = len(nums)
        while fast < numLen:
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
    
    # 28
    def strStr(self, haystack, needle):
        '''
        Function Description:
            find the start index where a substring in haystack matches needle

        Input:
            haystack: a string
            needle: a string

        Output:
            an int, the index from where a sub string matches with needle
        '''
        hayStackLen = len(haystack)
        needleLen = len(needle)
        for i in range(0, hayStackLen-needleLen+1):
            if haystack[i:i+needleLen] == needle:
                return i
        return -1
    
    # 35
    def searchInsert(self, nums, target):
        '''
        Function Description:
            find the index where the target should be inserted in sorted place

        Input:
            nums: a sorted array
            target: the int to be inserted

        Output:
            the insert index
        '''
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return left
    
    # 58
    def lengthOfLastWord(self, s):
        '''
        Function Description:
            find the length of the last word in the given string
        Input:
            s: a string of words with spaces at both ends

        Output:
            int, the length of the last word
        '''
        last = len(s) - 1
        cur = s[last]
        # go through the spaces at the end
        while cur == ' ' and last > -1:
            last -= 1
            cur = s[last]
        # count the length
        length = 0
        while cur != ' ' and last > -1:
            length += 1
            last -= 1
            cur = s[last]
        return length
    
    # 66
    def plusOne(self, digits):
        '''
        Function Description:
            simulate an adder, add 1 to a number

        Input:
            digits: the digits of a number in a list

        Output:
            list, riginal number + 1 in the form of a list of its digits
        '''
        i = len(digits) - 1
        carray = (digits[i] + 1) // 10
        digits[i] = (digits[i] + 1) % 10
        while carray != 0:
            # left most digit reached
            if i == 0:
                digits.insert(0, carray)
                break
            # carray out to the left digit
            i -= 1
            temp = digits[i] + carray
            digits[i] = temp % 10
            carray = temp // 10
        return digits

    # 67    
    def addBinary(self, a, b):
        '''
        Function Description:
            simulate a binary adder to add 2 binary numbers

        Input:
            a, b: the strings representing the 2 binary numbers

        Output:
            a string representing the sum
        '''
        binSum = []
        aIndex = len(a) - 1
        bIndex = len(b) - 1
        carray = 0
        # calculate
        while aIndex > -1 or bIndex > -1:
            if aIndex == -1:
                temp = int(b[bIndex]) + carray
                bIndex -= 1
            elif bIndex == -1:
                temp = int(a[aIndex]) + carray
                aIndex -= 1
            else:
                temp = int(a[aIndex]) + int(b[bIndex]) + carray
                aIndex -= 1
                bIndex -= 1
            curDigit = temp % 2
            carray = temp // 2
            binSum.append(curDigit)
        if carray > 0:
            binSum.append(carray)
        # reverse
        left = 0
        right = len(binSum) - 1
        while left <= right:
            temp = str(binSum[left])
            binSum[left] = str(binSum[right])
            binSum[right] = temp
            left += 1
            right -= 1
        return ''.join(binSum)
    
    # 69
    def mySqrt(self, x):
        '''
        Function Description:
            calculate the arithmetic square root of the input number

        Input:
            x: a non-negative int

        Output:
            the integer part of the sqrt
        '''
        left = 0
        right = x
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if mid ** 2 <= x:
                ans = mid
                left = mid + 1
            elif mid ** 2 > x:
                right = mid - 1
        return ans

    # 70
    def climbStairs(self, n):
        memo = [1, 2]
        i = 2
        while i <= n:
            memo.append(memo[i-1] + memo[i-2])
            i += 1
        return memo[n-1]
    
    # 74
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        nRows = len(matrix)
        nCols = len(matrix[0])
        numOfItems = nRows * nCols
        left = 0
        right = numOfItems - 1
        while left <= right:
            mid = (left + right) // 2
            # resolve to matrix coordinates
            midR = mid // nCols
            midC = mid % nCols
            if matrix[midR][midC] == target:
                return True
            elif matrix[midR][midC] > target:
                right = mid - 1
            else:
                left=  mid + 1
        return False
    
    # 83
    def deleteDuplicates(self, head):
        '''
        Function Description:
            delete duplicated elements in a single linked list

        Input:
            head: the head of a linked list

        Output:
            the head of the linked list with duplicates removed
        '''
        cur = head
        if head == None:
            return head
        while cur.next != None:
            if cur.next.val == cur.val:
                temp = cur.next
                while temp.val == cur.val and temp:
                    temp = temp.next
                    if temp == None:
                        break
                cur.next = temp
            else:
                cur = cur.next
        return head
    
    # 88
    def merge(self, nums1, m, nums2, n):
        '''
        Function Description:
            merge 2 arrays

        Input:
            nums1: array 1
            nums2: array 2
            m: length of array 1
            n: length of array 2

        Output:
            None
        '''
        for num in nums2:
            nums1[m] = num
            m += 1
        nums1.sort()
    
    # 94
    def inorderTraversal(self, root):
        stack = []
        ans = []
        while len(stack) != 0 or root != None:
            while root != None:
                stack.append(root)
                root = root.left
            root = stack.pop()
            ans.append(root.val)
            root = root.right
        return ans
    
    # 100
    def isSameTree(self, p, q):
        '''
        Function Description:
            determine if 2 trees are the same

        Input:
            p: the head of tree 1
            q: the head of tree 2

        Output:
            bool
        '''
        if p == None or q == None:
            if p == None and q == None:
                return True
            else:
                return False
        if p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    # 101
    def isSymmetric(self, root):
        '''
        Function Description:
            determine if a tree is axisymmetric around its root

        Input:
            root: the root node of the tree

        Output:
            bool
        '''
        return self.isMirror(root, root)
        
    def isMirror(self, left, right):
        '''
        Function Description:
            determine if a tree is mirrored

        Input:
            left: left node
            right: right node

        Output:
            bool
        '''
        if left == None and right == None:
            return True
        if left == None or right == None:
            return False
        return left.val == right.val and self.isMirror(left.left, right.right) and self.isMirror(left.right, right.left)

    # 104
    def maxDepth(self, root):
        '''
        Function Description:
            determine the max depth of a binary tree

        Input:
            root: the root node of the tree

        Output:
            int, the max depth
        '''
        if root == None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
    
    # 108
    def sortedArrayToBST(self, nums):
        
        def findRoot(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = findRoot(left, mid - 1)
            root.right = findRoot(mid + 1, right)
            return root

        return findRoot(0, len(nums) - 1)
    
    # 110
    def isBalanced(self, root):
        def height(root):
            if root == None:
                return 0
            lefth = height(root.left)
            righth = height(root.right)
            if lefth == -1 or righth == -1 or abs(righth - lefth) > 1:
                return -1
            else:
                return max(lefth, righth) + 1
        return height(root) >= 0
    
    # 111
    def minDepth(self, root):
        if root == None:
            return 0
        if root.left == None and root.right == None:
            return 1
        
        minDepth = 10 ** 5
        if root.left != None:
            minDepth = min(self.minDepth(root.left), minDepth)
        if root.right != None:
            minDepth = min(self.minDepth(root.right), minDepth)
        return minDepth + 1
    
    # 112
    def hasPathSum(self, root, targetSum):
        def helper(root, curSum):
            if root == None:
                return False
            curSum += root.val
            if curSum == targetSum and root.left == None and root.right == None:
                return True
            return helper(root.left, curSum) or helper(root.right, curSum)
        if root == None:
            return False
        return helper(root, 0) 
    
    # 118
    def generate(self, numRows):
        ans = []
        for i in range(numRows):
            row = []
            for j in range(0, i + 1):
                if j == 0 or j == i:
                    row.append(1)
                else:
                    row.append(ans[i-1][j-1] + ans[i-1][j])
            ans.append(row)
        return ans
    
    # 119
    def getRow(self, rowIndex):
        return self.generate(rowIndex+1)[rowIndex]
    
    # 121
    def maxProfit(self, prices):
        minPrice = prices[0]
        maxProfit = 0
        length = len(prices)
        i = 0
        while i < length:
            if prices[i] < minPrice:
                minPrice = prices[i]
            elif prices[i] - minPrice > maxProfit:
                maxProfit = prices[i] - minPrice
            i += 1
        return maxProfit

    # 125
    def isPalindrome(self, s):
        left = 0
        right = len(s) - 1
        while left <= right:
            # not a letter or a num
            if not s[left].isalnum():
                left += 1
                continue
            if not s[right].isalnum():
                right -= 1
                continue
            # compare
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        return True
    
    # 136
    def singleNumber(self, nums):
        if len(nums) < 2:
            return nums[0]
        discovered = {}
        for num in nums:
            if not num in discovered:
                discovered[num] = 1
            else:
                discovered[num] = discovered[num] + 1
        for key, val in discovered.items():
            if val == 1:
                return key
            
    # 138
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        def getNodes(head, nodes):
            if not head:
                return 
            newNode = Node(head.val)
            nodes.append(newNode)
            getNodes(head.next, nodes)

        def getOriginalNodes(head, nodes):
            if not head:
                return 
            nodes.append(head)
            getOriginalNodes(head.next, nodes)

        def getRandom(head, randoms, nodes):
            if not head:
                return 
            randoms.append(nodes.index(head.random) if head.random else None)
            getRandom(head.next, randoms, nodes)

        if not head:
            return None
        nodes = []
        getNodes(head, nodes)
        originalNodes = []
        getOriginalNodes(head, originalNodes)
        randoms = []
        getRandom(head, randoms, originalNodes)
        newHead = nodes[0]
        i = 0
        while i < len(nodes):
            nodes[i].next = nodes[i+1] if i != len(nodes) - 1 else None
            nodes[i].random = nodes[randoms[i]] if randoms[i] != None else None
            i += 1
        return newHead
    
    # 141
    def hasCycle(self, head):
        visited = {}
        while head:
            if head in visited:
                return True
            visited[head] = 1
            head = head.next
        return False
    
    # 144
    def preorderTraversal(self, root):
        def preorder(root):
            if root == None:
                return
            ans.append(root.val)
            preorder(root.left)
            preorder(root.right)
        
        ans = []
        preorder(root)
        return ans
    
    # 145
    def postorderTraversal(self, root):
        def postorder(root):
            if root == None:
                return
            postorder(root.left)
            postorder(root.right)
            ans.append(root.val)

        ans = []
        postorder(root)
        return ans
    
    # 151
    def reverseWords(self, s: str) -> str:
        def getWords(s: str) -> List[str]:
            words = []
            length = len(s)
            i = 0
            while i < length:
                while i < length and s[i] == " ":
                    i += 1
                j = i
                while j < length and s[j] != " ":
                    j += 1
                curWord = s[i:j]
                if curWord != "":
                    words.append(curWord)
                i = j
            return words
        
        s = getWords(s)
        return ' '.join(s[::-1])
    
    # 153
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        minNum = 5000
        while left <= right:
            if nums[left] < nums[right]:
                minNum = min(nums[left], minNum)
                break
            mid = (left + right) // 2
            minNum = min(minNum, nums[mid])
            if nums[mid] >= nums[left]:
                left = mid + 1
            else:
                right = mid - 1
        return minNum

    # 160
    def getIntersectionNode(self, headA, headB):
        visited = {}
        cur = headA
        while cur:
            visited[cur] = True
            cur = cur.next
        cur = headB
        while cur:
            if cur in visited:
                return cur
            cur = cur.next
        return None
    
    # 167
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while left < right:
            curSum = numbers[left] + numbers[right]
            if  curSum == target:
                return [left + 1, right + 1]
            if curSum > target:
                right -= 1
            else:
                left += 1
        return []
    
    # 168
    def convertToTitle(self, columnNumber):
        ans = []
        while columnNumber > 0:
            if columnNumber % 26 == 0:
                name = 'Z'
                columnNumber = columnNumber // 26 - 1
            else:
                name = chr(columnNumber % 26 + 64)
                columnNumber = columnNumber // 26
            ans.append(name)
            
        return ''.join(ans[::-1])

    # 169
    def majorityElement(self, nums):
        length = len(nums)
        counts = {}
        for num in nums:
            if num in counts:
                counts[num] = counts[num] + 1
            else:
                counts[num] = 1
        for key, val in counts.items():
            if val > math.floor(length / 2):
                return key
        return None
    
    # 171
    def titleToNumber(self, columnTitle):
        i = len(columnTitle) - 2
        ans = ord(columnTitle[-1]) - 64
        count = 1
        while i > -1:
            ans += (ord(columnTitle[i]) - 64) * count * 26
            count *= 26
            i -= 1
        return ans
    
    # 202
    def digitSum(self, n):
        ans = 0
        while n > 0:
            ans += (n % 10) ** 2
            n = n // 10
        return ans

    def isHappy(self, n):
        history = {n:1}
        curSum = n
        while curSum != 1:
            curSum = self.digitSum(n)
            if curSum in history:
                return False
            else:
                history[curSum] = 1
            n = curSum
        return True
    
    # 203
    def removeElements(self, head, val):
        while head:
            if head.val == val:
                head = head.next
            else:
                break
        cur = head
        while cur and cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head
    
    # 205
    def isIsomorphic(self, s, t):
        length = len(s)
        if length != len(t):
            return False
        dictMapS = {}
        dictMapT = {}
        for i in range(length):
            curMap = ord(s[i]) - ord(t[i])
            if s[i] in dictMapS:
                if chr(ord(s[i]) - dictMapS[s[i]]) != t[i]:
                    return False
            else:
                dictMapS[s[i]] = curMap
        for i in range(length):
            curMap = ord(s[i]) - ord(t[i])
            if t[i] in dictMapT:
                if chr(ord(t[i]) + dictMapT[t[i]]) != s[i]:
                    return False
            else:
                dictMapT[t[i]] = curMap
        return True
        
    # 206
    def reverseList(self, head):
        lastNode = None
        curNode = head
        while curNode:
            nextNode = curNode.next
            curNode.next = lastNode
            lastNode = curNode
            curNode = nextNode
        return lastNode
        
    # 217
    def containsDuplicate(self, nums):
        numsCount = {}
        for num in nums:
            if num in numsCount:
                return True
            numsCount[num] = 1
        return False
    
    # 219
    def containsNearbyDuplicate(self, nums, k):
        numIndex = {}
        length = len(nums)
        i = 0
        while i < length:
            curNum = nums[i]
            if curNum in numIndex:
                if abs(i - numIndex[curNum]) <= k:
                    return True
            numIndex[curNum] = i
            i += 1
        return False

    # 222
    def countNodes(self, root):
        if not root:
            return 0
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)
    
    # 226
    def invertTree(self, root):
        def flip(root):
            if root == None:
                return
            temp = root.left
            root.left = root.right
            root.right = temp
            flip(root.left)
            flip(root.right)

        temp = root
        flip(temp)
        return root
    
    # 228
    def summaryRanges(self, nums):
        if len(nums) == 0:
            return []
        ans = []
        left = 0
        right = left + 1
        while right < len(nums):
            temp = nums[right-1]
            # a range ends
            if nums[right] - temp != 1:
                # if range or single num
                if nums[left] != temp:
                    ans.append(str(nums[left]) + "->" + str(temp))
                else:
                    ans.append(str(temp))
                left = right
            right += 1
        # append the last range
        if nums[left] != nums[right-1]:
            ans.append(str(nums[left]) + "->" + str(nums[right-1]))
        else:
            ans.append(str(nums[right-1]))
        return ans
    
    # 231
    def isPowerOfTwo(self, n):
        if n <= 0:
            return False
        num = math.log2(n)
        return num == int(num)
    
    # 234
    def isPalindrome(self, head):
        nums = []
        length = 0
        while head:
            nums.append(head.val)
            length += 1
            head = head.next
        left = 0
        right = length - 1
        while left <= right:
            if nums[left] != nums[right]:
                return False
            left += 1
            right -= 1
        return True

    # 242
    def isAnagram(self, s, t):
        counts = {}
        for char in s:
            if char in counts:
                counts[char] = counts[char] + 1
            else:
                counts[char] = 1
        for char in t:
            if not char in counts:
                return False
            counts[char] = counts[char] - 1
        for key, val in counts.items():
            if val != 0:
                return False
        return True
    
    # 257
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        def constructPath(path: List[int]) -> str:
            return "->".join([str(p) for p in path])

        def getPaths(root: Optional[TreeNode], curPath: List[int]):
            if not root.left and not root.right:
                paths.append(curPath + [root.val])
            if root.left:
                getPaths(root.left, curPath + [root.val])
            if root.right:
                getPaths(root.right, curPath + [root.val])

        paths = []
        getPaths(root, [])
        return [constructPath(path) for path in paths]
    
    # 258
    def addDigits(self, num):
        def sumDigits(num: int) -> int:
            if num == 0:
                return 0
            return num % 10 + sumDigits(num // 10)
        
        while num // 10 != 0:
            num = sumDigits(num)
        return num
            
    # 263
    def isUgly(self, n):
        if n <= 0:
            return False
        while n > 0:
            if n == 1:
                return True
            if n % 2 == 0:
                n = n / 2
            elif n % 3 == 0:
                n = n / 3
            elif n % 5 == 0:
                n = n / 5
            else:
                return False
            
    # 268
    def missingNumber(self, nums):
        numsSet = set(nums)
        i = 0
        for num in numsSet:
            if i != num:
                return i
            i += 1

        return i
        
    # 278
    def firstBadVersion(self, n):
        def isBadVersion(n):
            pass

        if n < 2:
            return n
        left = 1
        right = n
        while left < right:
            mid = (left + right) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left
    
    # 283
    def moveZeroes(self, nums):
        lastPos = len(nums) - 1
        i = 0
        while i <= lastPos:
            if nums[i] == 0:
                j = i
                while j < lastPos:
                    nums[j] = nums[j+1]
                    j += 1
                nums[lastPos] = 0
                lastPos -= 1
            else:
                i += 1
    
    # 287
    def findDuplicate(self, nums: List[int]) -> int:
        numsSet = set()
        for num in nums:
            if num in numsSet:
                return num
            numsSet.add(num)
    
    # 290
    def wordPattern(self, pattern, s):
        patternDict = {}
        words = s.split(' ')
        if len(pattern) != len(words):
            return False
        i = 0
        while i < len(words):
            if pattern[i] in patternDict:
                if patternDict[pattern[i]] != words[i]:
                    return False
            else:
                patternDict[pattern[i]] = words[i]
            i += 1
        i = 0
        patternDict = {}
        while i < len(words):
            if words[i] in patternDict:
                if patternDict[words[i]] != pattern[i]:
                    return False
            else:
                patternDict[words[i]] = pattern[i]
            i += 1
        return True
    
    # 292
    def canWinNim(self, n):
        return n % 4 != 0
    
    # 299
    def getHint(self, secret: str, guess: str) -> str:
        secretDict = {}
        lengthS = len(secret)
        # preprocess to store the nums and their index
        i = 0
        while i < lengthS:
            cur = secret[i]
            if cur in secretDict:
                secretDict[cur].add(i)
            else:
                secretDict[cur] = {i}
            i += 1
        lengthG = len(guess)
        i = 0
        a = 0
        countB = [0] * 10
        # find match
        while i < lengthG:
            cur = guess[i]
            if cur in secretDict:
                if i in secretDict[cur]:
                    a += 1
                    secretDict[cur].remove(i)
                else:
                    countB[int(cur)] += 1
            i += 1
        b = 0
        i = 0
        while i < 10:
            temp = 0
            if str(i) in secretDict:
                temp = len(secretDict[str(i)])
            b += min(countB[i], temp)
            i += 1

        return str(a) + "A" + str(b) + "B"
    
    # 326
    def isPowerOfThree(self, n):
        if n <= 0:
            return False
        if n == 1:
            return True
        if n % 3 != 0:
            return False
        return self.isPowerOfThree(n / 3)

    # 342
    def isPowerOfFour(self, n):
        if n <= 0:
            return False
        if n == 1:
            return True
        if n % 4 != 0:
            return False
        return self.isPowerOfFour(n / 4)
    
    # 344
    def reverseString(self, s: List[str]) -> None:
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
    
    # 345
    def reverseVowels(self, s: str) -> str:
        left = 0
        right = len(s) - 1
        s = list(s)
        vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        while left < right:
            if s[left] in vowels and s[right] in vowels:
                s[left], s[right] = s[right], s[left]
                right -= 1
                left += 1
            elif s[left] in vowels:
                right -= 1
            elif s[right] in vowels:
                left += 1
            else:
                right -= 1
                left += 1
        return "".join(s)
    
    # 349
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        length1 = len(nums1)
        length2 = len(nums2)
        bigDict = {}
        if length1 > length2:
            for num in nums1:
                if not num in bigDict:
                    bigDict[num] = 0
            for num in nums2:
                if num in bigDict:
                    bigDict[num] = 1
        else:
            for num in nums2:
                if not num in bigDict:
                    bigDict[num] = 0
            for num in nums1:
                if num in bigDict:
                    bigDict[num] = 1
        intersects = []
        for key, val in bigDict.items():
            if val == 1:
                intersects.append(key)
        return intersects
    
    # 350
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dict1 = {}
        dict2 = {}
        # fill the dicts
        for num in nums1:
            if num in dict1:
                dict1[num] = dict1[num] + 1
            else:
                dict1[num] = 1
        for num in nums2:
            if num in dict2:
                dict2[num] = dict2[num] + 1
            else:
                dict2[num] = 1
        # find intersects
        intersects = []
        for key in dict2.keys():
            if key in dict1:
                for i in range(min(dict1[key], dict2[key])):
                    intersects.append(key)
        return intersects
    
    # 367
    def isPerfectSquare(self, num: int) -> bool:
        left = 1
        right = num
        while left <= right:
            mid = (left + right) // 2
            if mid * mid == num:
                return True
            if mid * mid > num:
                right = mid - 1
            else:
                left = mid + 1
        return False
    
    # 374
    def guessNumber(self, n: int) -> int:
        def guess(num: int) -> int:
            pass
        left  = 1
        right = n
        while left <= right:
            mid = (left + right) // 2
            match guess(mid):
                case 0:
                    return mid
                case 1:
                    left = mid + 1
                case -1:
                    right = mid - 1
        return -1
    
    # 383
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        charsCount = {}
        for char in magazine:
            if char in charsCount:
                charsCount[char] = charsCount[char] + 1
            else:
                charsCount[char] = 1
        for char in ransomNote:
            if not char in charsCount:
                return False
            elif charsCount[char] < 1:
                return False
            charsCount[char] = charsCount[char] - 1
        return True

    # 387
    def firstUniqChar(self, s: str) -> int:
        charsCount = {}
        for char in s:
            if char in charsCount:
                charsCount[char] = charsCount[char] + 1
            else:
                charsCount[char] = 1
        i = 0
        while i < len(s):
            if charsCount[s[i]] == 1:
                return i
            i += 1
        return -1

    # 389
    def findTheDifference(self, s: str, t: str) -> str:
        charsCount = {}
        for char in t:
            if char in charsCount:
                charsCount[char] = charsCount[char] + 1
            else:
                charsCount[char] = 1
        for char in s:
            charsCount[char] = charsCount[char] - 1
        for key, val in charsCount.items():
            if val == 1:
                return str(key)
        return ""
    
    # 392
    def isSubsequence(self, s: str, t: str) -> bool:
        if s == "":
            return True
        sIndex = 0
        tIndex = 0
        while sIndex < len(s):
            while tIndex < len(t):
                if s[sIndex] == t[tIndex]:
                    tIndex += 1
                    if sIndex == len(s) - 1:
                        return True
                    break
                tIndex += 1
            sIndex += 1
        return False
    
    # 401
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        ans = []
        for h in range(12):
            for m in range(60):
                if bin(h).count('1') + bin(m).count(1) == turnedOn:
                    ans.append(str(h) + ":" + f"{m:02d}")
        return ans
    
    # 404
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        def isLeaf(node):
            return (not node.left) and (not node.right)
        
        if root == None:
            return 0
        if root.left:
            if isLeaf(root.left):
                return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)
    
    # 405
    def toHex(self, num: int) -> str:
        if num < 0:
            num = (1<<32) + num
        return hex(num)[2:]
    
    # 409
    def longestPalindrome(self, s: str) -> int:
        counts = {}
        for char in s:
            if char in counts:
                counts[char] = counts[char] + 1
            else:
                counts[char] = 1
        oddFound = False
        ans = 0
        for val in counts.values():
            if val % 2 == 0:
                ans += val
            else:
                ans += val - 1
                oddFound = True
        if oddFound:
            ans += 1
        return ans
    
    # 412
    def fizzBuzz(self, n: int) -> List[str]:
        answer = []
        for i in range(1, n+1):
            if i % 3 == 0 and i % 5 == 0:
                answer.append("FizzBuzz")
            elif i % 3 == 0:
                answer.append("Fizz")
            elif i % 5 == 0:
                answer.append("Buzz")
            else:
                answer.append(str(i))
        return answer
    
    # 414
    def thirdMax(self, nums: List[int]) -> int:
        nums.sort()
        length = len(nums)
        count = 1
        i = length - 1
        cur = nums[i]
        while i > -1 and count < 3:
            if nums[i] != cur:
                count += 1
                cur = nums[i]
            i -= 1
        if count == 3:
            return nums[i+1]
        else:
            return nums[length-1]
    
    # 415
    def addStrings(self, num1: str, num2: str) -> str:
        ans = []
        index1 = len(num1) - 1
        index2 = len(num2) - 1
        carry = 0
        while index1 > -1 or index2 > -1:
            n1 = int(num1[index1])
            n2 = int(num2[index2])
            if index1 > -1 and index2 > -1:
                ans.append((carry + n1 + n2) % 10)
                carry = (carry + n1 + n2) // 10
                index1 -= 1
                index2 -= 1
            elif index1 > -1:
                ans.append((n1 + carry) % 10)
                carry = (n1 + carry) // 10
                index1 -= 1
            else:
                ans.append((n2 + carry) % 10)
                carry = (n2 + carry) // 10
                index2 -= 1
        if carry != 0:
            ans.append(carry)
        return ''.join(map(str, ans[::-1]))
    
    # 424
    def characterReplacement(self, s: str, k: int) -> int:
        counts = {}
        maxLen = 0
        left = 0
        right = 0
        while right < len(s):
            counts[s[right]] = 1 + counts.get(s[right], 0)
            while right - left + 1 - max(counts.values()) > k:
                counts[s[left]] = max(0, counts[s[left]] - 1)
                left += 1
            maxLen = max(maxLen, right - left + 1)
            right += 1
        return maxLen
    
    # 434
    def countSegments(self, s: str) -> int:
        isSeg = False
        count = 0
        length = len(s)
        i = 0
        while i < length:
            if s[i] != ' ':
                isSeg = True
            if (s[i] == ' ' or i == length - 1) and isSeg:
                count += 1
                isSeg = False
            elif s[i] == ' ':
                isSeg = False
            i += 1
        return count
    
    # 441
    def arrangeCoins(self, n: int) -> int:
        def getCoins(rows):
            return (1 + rows) * rows / 2
        
        left = 1
        right = n
        while left <= right:
            mid = (left + right) // 2
            if getCoins(mid) == n:
                return mid
            elif getCoins(mid) > n:
                right = mid - 1
            else:
                left = mid + 1
        return right
    
    # 443
    def compress(self, chars: List[str]) -> int:
        def numDigits(num: int) -> List[str]:
            digits = []
            while num > 0:
                digits.append(str(num % 10))
                num //= 10
            return digits[::-1]

        count = 1
        cur = chars[0]
        res = []
        i = 1
        while i < len(chars):
            if chars[i] != cur:
                res.append(cur)
                if count > 1:
                    res += numDigits(count)
                count = 1
                cur = chars[i]
            else:
                count += 1
            i += 1
        res.append(cur)
        if count > 1:
            res += numDigits(count)
        i = 0
        while i < len(res):
            chars[i] = res[i]
            i += 1

        return len(res)
    
    # 448
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        present = [False] * len(nums)
        for num in nums:
            present[num-1] = True
        ans = []
        i = 0
        while i < len(present):
            if not present[i]:
                ans.append(i + 1)
            i += 1
        return ans
    
    # 455
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        gIndex = len(g) - 1
        sIndex = len(s) - 1
        count = 0
        while gIndex > -1 and sIndex > -1:
            if s[sIndex] >= g[gIndex]:
                count += 1
                sIndex -= 1
                gIndex -= 1
            else:
                gIndex -= 1
        return count
    
    # 459
    def repeatedSubstringPattern(self, s: str) -> bool:
        length = len(s)
        if length < 2:
            return False
        for i in range(1, length // 2 + 1):
            if s[:i] * (length // i) == s:
                return True
        return False
    
    # 461
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count(1)
    
    # 463
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        length = len(grid)
        width = len(grid[0])
        count = 0
        for i in range(length):
            for j in range(width):
                if grid[i][j] == 1:
                    if j == 0 or grid[i][j - 1] == 0:
                        count += 1
                    if i == 0 or grid[i - 1][j] == 0:
                        count += 1
                    if j == len(grid[i]) - 1 or grid[i][j + 1] == 0:
                        count += 1
                    if i == len(grid) - 1 or grid[i + 1][j] == 0:
                        count += 1
        return count

    # 476
    def findComplement(self, num: int) -> int:
        numBin = bin(num)[2:]
        ans = []
        for i in numBin:
            ans.append('1' if i == '0' else '0')
        return int(''.join(ans), 2)
    
    # 482
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        i = len(s) - 1
        count = 0
        keys = []
        while i > -1:
            if s[i] == '-':
                i -= 1
                continue
            if count == k:
                keys.append('-')
                count = 0
            keys.append(s[i].upper())
            count += 1
            i -= 1
        return ''.join(keys[::-1])
    
    # 485
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        curCount = 0
        maxCount = 0
        for i in nums:
            if i == 0:
                curCount = 0
            else:
                curCount += 1
            if curCount > maxCount:
                maxCount = curCount
        return maxCount
    
    # 492
    def constructRectangle(self, area: int) -> List[int]:
        width = int(math.sqrt(area))
        while area % width != 0:
            width -= 1
        return [area // width, width]
    
    # 495
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        i = 0
        ans = 0
        while i < len(timeSeries):
            if i == len(timeSeries) - 1:
                ans += duration
            elif timeSeries[i] + duration > timeSeries[i+1]:
                ans += timeSeries[i+1] - timeSeries[i]
            else:
                ans += duration
            i += 1
        return ans
    
    # 496
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # store the next greatest numbers in nums2
        record = []
        i = 0
        while i < len(nums2):
            j = i + 1
            found = False
            while j < len(nums2):
                if nums2[j] > nums2[i]:
                    record.append(nums2[j])
                    found = True
                    break
                j += 1
            if not found:
                record.append(-1)
            i += 1
        # match nums1 and nums2
        ans = []
        index1 = 0
        while index1 < len(nums1):
            index2 = 0
            while index2 < len(nums2):
                if nums1[index1] == nums2[index2]:
                    found = True
                    ans.append(record[index2])
                    break
                index2 += 1
            index1 += 1
        return ans
    
    # 500
    def strToList(self, s: str) -> List[str]:
        returnList = []
        for char in s:
            returnList.append(char)
        return returnList

    def findWords(self, words: List[str]) -> List[str]:
        rowIndex = "12210111011122000010020202"
        ans = []
        for word in words:
            curIndex = rowIndex[ord(word[0].lower()) - ord('a')]
            i = 1
            isValid = True
            while i < len(word):
                if curIndex != rowIndex[ord(word[i].lower()) - ord('a')]:
                    isValid = False
                    break
                i += 1
            if isValid:
                ans.append(word)
        return ans

    # 501
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        def getVals(root, valsCount):
            '''
            Count the occurance of each val
            '''
            if not root:
                return
            else:
                if root.val in valsCount:
                    valsCount[root.val] = valsCount[root.val] + 1
                else:
                    valsCount[root.val] = 1
                getVals(root.left, valsCount)
                getVals(root.right, valsCount)
        valsCount = {}
        getVals(root, valsCount)
        # find max occurance
        maxOccur = valsCount[root.val]
        for val in valsCount.values():
            if val > maxOccur:
                maxOccur = val
        # find mode
        modes = []
        for key, val in valsCount.items():
            if val == maxOccur:
                modes.append(key)
        return modes
    
    # 504
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return "0"
        isNegative = False
        if num < 0:
            isNegative = True
            num = -num
        base10 = []
        while num > 0:
            base10.append(num % 7)
            num = num // 7
        if isNegative:
            base10.append('-')
        return ''.join(map(str, base10[::-1]))
    
    # 506
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        # store the original index
        originalOrder = {}
        i = 0
        while i < len(score):
            originalOrder[score[i]] = i
            i += 1
        # assign ranks
        score.sort(reverse=True)
        answer = [None] * len(score)
        i = 1
        for x in score:
            index = originalOrder[x]
            match i:
                case 1:
                    answer[index] = "Gold Medal"
                case 2:
                    answer[index] = "Silver Medal"
                case 3:
                    answer[index] = "Bronze Medal"
                case _:
                    answer[index] = str(i)
            i += 1
        return answer
    
    # 507
    def checkPerfectNumber(self, num: int) -> bool:
        if num == 1:
            return False
        factorSum = 1
        i = 2
        while i <= math.sqrt(num):
            if num % i == 0:
                factorSum += i
                if i * i < num:
                    factorSum += num // i
            i += 1
        return factorSum == num
    
    # 509
    def fib(self, n: int) -> int:
        # calc memo
        fibs = [0, 1]
        for i in range(2, n + 1):
            fibs.append(fibs[i-2] + fibs[i-1])
        return fibs[n]
    
    # 520
    def detectCapitalUse(self, word: str) -> bool:
        if len(word) < 2:
            return True
        firstUpper = word[0].isupper()
        secondUpper = word[1].isupper()
        if not firstUpper and secondUpper:
            return False
        for i in range(2, len(word)):
            if word[i].isupper() != secondUpper:
                return False
        return True
    
    # 530
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        def getVals(root, vals):
            if not root:
                return
            vals.append(root.val)
            getVals(root.left, vals)
            getVals(root.right, vals)
        vals = []
        getVals(root, vals)
        minDif = 10 ** 5
        for val1 in vals:
            for val2 in vals:
                if abs(val1 - val2) != 0 and abs(val1 - val2) < minDif:
                    minDif = abs(val1 - val2)
        return minDif
    
    # 541
    def reverseStr(self, s: str, k: int) -> str:
        def reverse(l: List[int], left: int, right: int):
            while left <= right:
                l[left], l[right] = l[right], l[left]
                left += 1
                right -= 1
        sList = []
        for i in s:
            sList.append(i)
        left = 0
        right = 0
        count = 0
        while right < len(sList):
            count += 1
            if count == 2 * k:
                count = 0
                reverse(sList, left, right - k)
                left = right + 1
            right += 1
        # handle leftovers
        if count >= k and count < 2 * k:
            reverse(sList, left, left + k - 1)
        elif count < k:
            reverse(sList, left, len(sList) - 1)
        return ''.join(map(str, sList))
    
    # 543
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def getDepth(root):
            if not root:
                return 0
            l = getDepth(root.left)
            r = getDepth(root.right)
            self.depth = max(l + r + 1, self.depth)
            return max(l, r) + 1
        self.depth = 1
        getDepth(root)
        return self.depth - 1
    
    # 551
    def checkRecord(self, s: str) -> bool:
        absentCount = 0
        contLateCount = 0
        late = False
        for stat in s:
            if stat == 'A':
                absentCount += 1
                contLateCount = 0
            elif stat == 'L':
                contLateCount += 1
                if contLateCount >= 3:
                    late = True
            else:
                contLateCount = 0
            if late or absentCount >= 2:
                return False
        return not (late or absentCount >= 2)
    
    # 557
    def reverseWords(self, s: str) -> str:
        def reverse(l: List[int], left: int, right: int):
            while left <= right:
                l[left], l[right] = l[right], l[left]
                left += 1
                right -= 1
        s = self.strToList(s)
        left = 0
        right = 0
        while right < len(s):
            if s[right] == ' ':
                reverse(s, left, right - 1)
                right += 1
                left = right
            else:
                right += 1
        reverse(s, left, right - 1)
        return ''.join(s)
    
    # 559
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        maxDepth = 0
        for child in root.children:
            childDepth = maxDepth(child)
            maxDepth = max(childDepth, maxDepth)
        return maxDepth + 1
    
    # 561
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        i = 0
        pairSum = 0
        while i < len(nums):
            if i % 2 != 0:
                pairSum += min(nums[i], nums[i-1])
            i += 1
        return pairSum
        
    # 563
    def findTilt(self, root: Optional[TreeNode]) -> int:
        def tilt(root):
            if not root:
                return 0
            lSum = tilt(root.left)
            rSum = tilt(root.right)
            self.ans += abs(lSum - rSum)
            return lSum + rSum + root.val
        self.ans=  0
        tilt(root)
        return self.ans
    
    # 566
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        if not len(mat) * len(mat[0]) == r * c:
            return mat
        ans = []
        temp = []
        count = 0
        for rows in mat:
            for num in rows:
                count += 1
                temp.append(num)
                if count % c == 0:
                    ans.append(temp)
                    temp = []
                    count = 0
        return ans
    
    # 567
    def checkInclusion(self, s1: str, s2: str) -> bool:
        # count chars in s1
        chars = {}
        for char in s1:
            chars[char] = chars.get(char, 0) + 1
        length1 = len(s1)
        length2 = len(s2)
        left = 0
        right = left
        # copy chars
        tempChars = {}
        for key, val in chars.items():
                tempChars[key] = val
        while right < length2:
            # length exceeded
            if right - left + 1 > length1:
                for key, val in chars.items():
                    tempChars[key] = val
                left += 1
                continue
            # not a match
            if not s2[right] in tempChars:
                left += 1
                for key, val in chars.items():
                    tempChars[key] = val
                right = left
            # match found
            else:
                tempChars[s2[right]] = tempChars[s2[right]] - 1
                if tempChars[s2[right]] < 1:
                    del tempChars[s2[right]]
                right += 1
                if not tempChars:
                    return True
        return False
    
    # 572
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def sameTree(root, root1):
            if not root and not root1:
                return True
            elif not root or not root1:
                return False
            if not root.val == root1.val:
                return False
            else:
                return sameTree(root.left, root1.left) and sameTree(root.right, root1.right)
        if not root and not subRoot:
            return True
        elif not root or not subRoot:
            return False
        if root.val != subRoot.val:
            return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        else:
            return sameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
        
    # 575
    def distributeCandies(self, candyType: List[int]) -> int:
        numOfCandies = len(candyType)
        typeCount = {}
        numOfTypes = 0
        for candy in candyType:
            if not candy in typeCount:
                numOfTypes += 1
                typeCount[candy] = 1
        if numOfTypes < numOfCandies // 2:
            return numOfTypes
        else:
            return numOfCandies // 2
    
    # 589
    def preorder(self, root: 'Node') -> List[int]:
        def preorderTraversal(root):
            if not root:
                return
            self.ans.append(root.val)
            for child in root.children:
                preorderTraversal(child)
        self.ans = []
        preorderTraversal(root)
        return self.ans
    
    # 590
    def postorder(self, root: 'Node') -> List[int]:
        def postorderTraversal(root):
            if not root:
                return
            for child in root.children:
                postorderTraversal(child)
            self.ans.append(root.val)
        self.ans = []
        postorderTraversal(root)
        return self.ans
    
    # 594
    def findLHS(self, nums: List[int]) -> int:
        numCount = {}
        for num in nums:
            if num in numCount:
                numCount[num] = numCount[num] + 1
            else:
                numCount[num] = 1
        maxCount = 0
        for num in nums:
            if num + 1 in numCount.keys():
                maxCount = max(maxCount, numCount[num] + numCount[num + 1])
        return maxCount
    
    # 598
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        minRow = m
        minCol = n
        for row, col in ops:
            if row < minRow:
                minRow = row
            if col < minCol:
                minCol = col
        return minRow * minCol
    
    # 599
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        restaurants = {}
        i = 0
        # store the restaurants in list1
        while i < len(list1):
            restaurants[list1[i]] = i
            i += 1
        i = 0
        minIndexSum = len(list1) + len(list2)
        # find the min index sum
        while i < len(list2):
            if list2[i] in restaurants.keys():
                if i + restaurants[list2[i]] < minIndexSum:
                    minIndexSum = i + restaurants[list2[i]]
            i += 1
        i = 0
        ans = []
        # find the restaurants they have in common with the min index sum
        while i < len(list2):
            if list2[i] in restaurants.keys():
                if i + restaurants[list2[i]] == minIndexSum:
                    ans.append(list2[i])
            i += 1
        return ans


# %%
import heapq

# 703
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.queue = nums
        heapq.heapify(self.queue)
        
    def add(self, val: int) -> int:
        heapq.heappush(self.queue, val)
        while len(self.queue) > self.k:
            heapq.heappop(self.queue)
        return self.queue[0]
    
    


# 705
class MyHashSet:

    def __init__(self):
        self.content = [None] * 101

    def add(self, key: int) -> None:
        hashIndex = key % 101
        if self.content[hashIndex] == None:
            self.content[hashIndex] = key
        elif type(self.content[hashIndex]) == int:
            if key != self.content[hashIndex]:
                self.content[hashIndex] = [self.content[hashIndex], key]
        else:
            if not key in self.content[hashIndex]:
                self.content[hashIndex].append(key)


    def remove(self, key: int) -> None:
        hashIndex = key % 101
        if self.content[hashIndex] == None:
            return
        elif type(self.content[hashIndex]) == int:
            if self.content[hashIndex] == key:
                self.content[hashIndex] = None
        else:
            i = 0
            while i < len(self.content[hashIndex]):
                if self.content[hashIndex][i] == key:
                    self.content[hashIndex].remove(key)
                    break
                else:
                    i += 1
                    

    def contains(self, key: int) -> bool:
        hashIndex = key % 101
        if self.content[hashIndex] == None:
            return False
        elif type(self.content[hashIndex]) == int:
            return self.content[hashIndex] == key
        else:
            return key in self.content[hashIndex]


# 706
class MyHashMap:

    def __init__(self):
        self.length = 211
        self.content = [None] * self.length

    def put(self, key: int, value: int) -> None:
        keyIndex = key % self.length
        # index not occupied
        if self.content[keyIndex] == None:
            self.content[keyIndex] = [[key, value]]
        # index occupied
        else:
            i = 0
            while i < len(self.content[keyIndex]):
                if self.content[keyIndex][i][0] == key:
                    self.content[keyIndex][i][1] = value
                    return
                i += 1
            self.content[keyIndex].append([key, value])
            

    def get(self, key: int) -> int:
        keyIndex = key % self.length
        if self.content[keyIndex] == None:
            return -1
        else:
            for k, val in self.content[keyIndex]:
                if k == key:
                    return val
            return -1

    def remove(self, key: int) -> None:
        keyIndex = key % self.length
        if self.content[keyIndex] != None:
            i = 0
            while i < len(self.content[keyIndex]):
                if self.content[keyIndex][i][0] == key:
                    del self.content[keyIndex][i]
                    break
                i += 1
            if len(self.content[keyIndex]) == 0:
                self.content[keyIndex] = None


class Solution:
    # 605
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0
        i = 0
        while i < len(flowerbed):
            if flowerbed[min(i + 1, len(flowerbed) - 1)] == 0 and flowerbed[max(i - 1, 0)] == 0 and flowerbed[i] == 0:
                count += 1
                flowerbed[i] = 1
            i += 1
        return count >= n

    # 606
    def tree2str(self, root: Optional[TreeNode]) -> str:
        if not root:
            return ""
        elif not root.left and not root.right:
            return f"{root.val}"
        elif not root.left:
            return f"{root.val}()({self.tree2str(root.right)})"
        elif not root.right:
            return f"{root.val}({self.tree2str(root.left)})"
        else:
            return f"{root.val}({self.tree2str(root.left)})({self.tree2str(root.right)})"
        
    # 617
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1:
            return root2
        if not root2:
            return root1
        temp = TreeNode(root1.val + root2.val)
        temp.left = self.mergeTrees(root1.left, root2.left)
        temp.right = self.mergeTrees(root1.right, root2.right)
        return temp
    
    # 628
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        return max(nums[0] * nums[1] * nums[n - 1], nums[n - 3] * nums[n - 2] * nums[n - 1])
    
    # 637
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        def dfs(root: TreeNode, level: int):
            if not root:
                return
            if level < len(totals):
                totals[level] += root.val
                counts[level] += 1
            else:
                totals.append(root.val)
                counts.append(1)
            dfs(root.left, level + 1)
            dfs(root.right, level + 1)

        counts = list()
        totals = list()
        dfs(root, 0)
        return [total / count for total, count in zip(totals, counts)]
    
    # 643
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        maxSum = sum(nums[:k])
        curSum = maxSum
        i = 1
        while i <= len(nums) - k:
            curSum = curSum - nums[i-1] + nums[i+k-1]
            if maxSum < curSum:
                maxSum = curSum
            i += 1
        return maxSum / k
    
    # 645
    def findErrorNums(self, nums: List[int]) -> List[int]:
        presence = []
        for i in range(len(nums)):
            presence.append(0)
        for num in nums:
            presence[num-1] += 1
        i = 0
        ans = [-1, -1]
        while i < len(presence):
            if presence[i] == 0:
                ans[1] = i+1
            if presence[i] > 1:
                ans[0] = i+1
            i += 1
        return ans
    
    # 648
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        words = sentence.split(" ")
        dictionary.sort(key=len)
        for i in range(len(words)):
            wordLen = len(words[i])
            for root in dictionary:
                rootLen = len(root)
                if rootLen <= wordLen and root == words[i][:rootLen]:
                    words[i] = root
                    break
        return " ".join(words)
    
    # 653
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        def getVals(root):
            if not root:
                return
            self.vals[root.val] = 1
            getVals(root.left)
            getVals(root.right)

        self.vals = {}
        getVals(root)
        for key in self.vals.keys() and k - key != key:
            if k - key in self.vals.keys():
                return True
        return False
    
    # 657
    def judgeCircle(self, moves: str) -> bool:
        coord = [0, 0]
        for move in moves:
            match move:
                case 'U':
                    coord[1] += 1
                case 'D':
                    coord[1] -= 1
                case 'L':
                    coord[0] -= 1
                case 'R':
                    coord[0] += 1
        return coord == [0, 0]
    
    # 661
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        i = 0
        j = 0
        lenU = len(img)
        lenV = len(img[0])
        ans = []
        while i < lenU:
            curRow = []
            while j < lenV:
                curSum = 0
                adjacentCount = 0
                # calc sum
                for u in range(i - 1, i + 2):
                    for v in range(j - 1, j + 2):
                        # index out of range
                        if u >= lenU or v >= lenV or u < 0 or v < 0:
                            continue
                        curSum += img[u][v]
                        adjacentCount += 1
                curRow.append(math.floor(curSum / adjacentCount))
                j += 1
            ans.append(curRow)
            i += 1
            j = 0
        return ans
    
    # 671
    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        def logVals(root):
            if not root:
                return
            if root.val < self.minVal:
                self.minVal = root.val
            if root.val not in self.vals:
                self.vals[root.val] = 1
            else:
                self.vals[root.val]  = self.vals[root.val] + 1
            logVals(root.left)
            logVals(root.right)
        # get all the values in the tree and the min value
        self.vals = {}
        self.minVal = root.val
        logVals(root)
        # only one value in tree
        if len(self.vals) == 1:
            return -1
        # remove min
        del self.vals[self.minVal]
        # find second min
        sndMin = 2 ** 31 - 1
        for val in self.vals.keys():
            if val < sndMin:
                sndMin = val
        return sndMin
    
    # 674
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        left = 0
        right = 1
        maxLen = 1
        while right < len(nums):
            if nums[right] > nums[right - 1]:
                maxLen = max(maxLen, right - left + 1)
            else:
                left = right
            right += 1
        return maxLen
    
    # 680
    def validPalindrome(self, s: str) -> bool:
        def isPalindrome(left, right, s):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        length = len(s)
        left = 0
        right = length - 1
        while left < right:
            if s[left] != s[right]:
                return isPalindrome(left + 1, right, s) or isPalindrome(left, right - 1, s)
            left += 1
            right -= 1
        return True
    
    # 682
    def calPoints(self, operations: List[str]) -> int:
        points = []
        i = 0
        pointLength = 0
        while i < len(operations):
            match operations[i]:
                case "+":
                    points.append(int(points[pointLength-1]) + int(points[pointLength-2]))
                    pointLength += 1
                case "D":
                    points.append(int(points[pointLength-1]) * 2)
                    pointLength += 1
                case "C":
                    points.pop()
                    pointLength -= 1
                case _:
                    points.append(int(operations[i]))
                    pointLength += 1
            i += 1
        return sum(points)
    
    # 693
    def hasAlternatingBits(self, n: int) -> bool:
        nBin = bin(n)[2:]
        isZero = False
        for i in nBin:
            if isZero and i != '0':
                return False
            if not isZero and i == '0':
                return False
            if i == '0':
                isZero = False
            else:
                isZero = True
        return True

    # 696
    def countBinarySubstrings(self, s: str) -> int:
        temp = s[0]
        counts = []
        count = 1
        for i in range(1, len(s)):
            if temp == s[i]:
                count += 1
            else:
                temp = s[i]
                counts.append(count)
                count = 1
        counts.append(count)
        i = 0
        count = 0
        while i < len(counts) - 1:
            count += min(counts[i], counts[i + 1])
            i += 1
        return count
    
    # 697
    def findShortestSubArray(self, nums: List[int]) -> int:
        counts = {}
        i = 0
        while i < len(nums):
            if nums[i] in counts:
                counts[nums[i]] = [counts[nums[i]][0], i, counts[nums[i]][2] + 1]
            else:
                counts[nums[i]] = [i, i, 1]
            i += 1
        degree = counts[nums[0]][2]
        minDif = counts[nums[0]][1] - counts[nums[0]][0]
        for val in counts.values():
            if val[2] > degree:
                degree = val[2]
                minDif = val[1] - val[0]
            elif val[2] == degree:
                minDif = min(minDif, val[1] - val[0])
        return minDif + 1
    
    # 700
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return
        if root.val == val:
            return root
        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    # 704
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1
    
    # 709
    def toLowerCase(self, s: str) -> str:
        sList = []
        i = 0
        while i < len(s):
            charOrd = ord(s[i])
            if charOrd > 64 and charOrd < 91:
                sList.append(chr(charOrd + 32))
            else:
                sList.append(s[i])
            i += 1
        return ''.join(sList)
    
    # 724
    def pivotIndex(self, nums: List[int]) -> int:
        # calc prefix sum
        memo = [0]
        length = len(nums)
        for i in range(0, length):
            memo.append(nums[i] + memo[i])
        # find pivot index
        pIndex = -1
        for i in range(0, length):
            if memo[i] == memo[length] - memo[i+1]:
                return i
        return pIndex
    
    # 728
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        def isSelfDividing(num):
            temp = num
            while temp > 0:
                curDigit = temp % 10
                if curDigit == 0 or num % curDigit != 0:
                    return False
                temp = temp // 10
            return True
        
        ans = []
        for i in range(left, right+1):
            if isSelfDividing(i):
                ans.append(i)
        return ans
    
    # 733
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        queue = [[sr, sc]]
        init = image[sr][sc]
        lengthSr = len(image)
        lengthSc = len(image[0])
        visited = {}
        while queue:
            cur = queue.pop(0)
            u = cur[0]
            v = cur[1]
            image[u][v] = color
            # skip if already visited
            if (u, v) in visited:
                continue
            # mark the current pixel visited
            visited[(u, v)] = 1
            # if the 4 pixels have value equals to the initial value, enqueue
            if u - 1 > -1:
                if image[u - 1][v] == init:
                    queue.append([u - 1, v])
            if u + 1 < lengthSr:
                if image[u + 1][v] == init:
                    queue.append([u + 1, v])
            if v - 1 > -1:
                if image[u][v - 1] == init:
                    queue.append([u, v - 1])
            if v + 1 < lengthSc:
                if image[u][v + 1] == init:
                    queue.append([u, v + 1])
        return image
    
    # 744
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        ans = 'z'
        found = False
        for char in letters:
            if char > target and char <= ans:
                ans = char
                found = True
        if not found:
            return letters[0]
        return ans
    
    # 746
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        memo = [0, 0]
        length = len(cost)
        for i in range(2, length + 1):
            memo.append(min(memo[i - 1] + cost[i - 1], memo[i - 2] + cost[i - 2]))
        return memo[length]
    
    # 747
    def dominantIndex(self, nums: List[int]) -> int:
        # get the index of the max number
        maxIndex = 0
        length = len(nums)
        i = 1
        while i < length:
            if nums[i] > nums[maxIndex]:
                maxIndex = i
            i += 1
        # determine if max number is at least twice as much as every number
        i = 0
        while i < length:
            if i == maxIndex:
                i += 1
                continue
            if nums[i] * 2 > nums[maxIndex]:
                return -1
            i += 1
        return maxIndex
    
    # 748
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        # count all the alphabets in licensePlate
        count = []
        for i in range(26):
            count.append(0)
        for char in licensePlate:
            if char.isalpha():
                char = char.lower()
                count[ord(char) - 97] += 1
        # find valid words
        validWords = []
        for word in words:
            curCount = []
            length = 0
            for i in range(26):
                curCount.append(0)
            for char in word:
                length += 1
                curCount[ord(char) - 97] += 1
            # check if is valid
            isValid = True
            for i in range(26):
                if curCount[i] < count[i]:
                    isValid = False
                    break
            if isValid:
                validWords.append((word, length))
        # find shortest
        minLen = validWords[0][1]
        ans = validWords[0][0]
        for word in validWords:
            if word[1] < minLen:
                minLen = word[1]
                ans = word[0]
        return ans
    
    # 762
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def isPrime(num):
            if num < 2:
                return False
            for i in range(2, num):
                if num % i == 0:
                    return False
            return True
        ans = 0
        for i in range(left, right + 1):
            if isPrime(bin(i).count('1')):
                ans += 1
        return ans
    
    # 766
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        numOfRows = len(matrix)
        numOfCols = len(matrix[0])
        # check upper half
        for i in range(numOfCols - 1):
            m = 0
            n = i
            temp = matrix[m][n]
            while m < numOfRows and n < numOfCols:
                if matrix[m][n] != temp:
                    return False
                m += 1
                n += 1
        # check for lower half
        for i in range(numOfRows - 1):
            m = i
            n = 0
            temp = matrix[m][n]
            while m < numOfRows and n < numOfCols:
                if matrix[m][n] != temp:
                    return False
                m += 1
                n += 1
        return True
    
    # 771
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        jewelList = set()
        for jewel in jewels:
            jewelList.add(jewel)
        count = 0
        for stone in stones:
            if stone in jewelList:
                count += 1
        return count
    
    # 783
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        def getVals(root):
            if not root:
                return
            getVals(root.left)
            self.vals.append(root.val)
            self.length += 1
            getVals(root.right)

        self.vals = []
        self.length = 0
        getVals(root)
        minDif = abs(self.vals[0] - self.vals[1])
        for i in range(1, self.length - 1):
            if abs(self.vals[i] - self.vals[i + 1]) < minDif:
                minDif = abs(self.vals[i] - self.vals[i + 1])
        return minDif
    
    # 796
    def rotateString(self, s: str, goal: str) -> bool:
        i = 0
        gLength = len(goal)
        sLength = len(s)
        if gLength != sLength:
            return False
        while i < gLength:
            # match begin
            if goal[i] == s[0]:
                sIndex = 0
                gIndex = i
                # check if the rest match
                while sIndex < sLength:
                    if s[sIndex] != goal[gIndex % gLength]:
                        break
                    if sIndex == sLength - 1:
                        return True
                    sIndex += 1
                    gIndex += 1
            i += 1
        return False

    # 804
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        def translate(word):
            dictionary = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
            translation = []
            for char in word:
                translation.append(dictionary[ord(char) - 97])
            return "".join(translation)
        
        translations = set()
        for word in words:
            translations.add(translate(word))
        ans = 0
        for i in translations:
            ans += 1
        return ans
        
    # 806
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        rows = 0
        count = 0
        i = 0
        length = len(s)
        while i < length:
            count += widths[ord(s[i]) - 97]
            if count == 100:
                count = 0
                rows += 1
            elif count > 100:
                count = 0
                rows += 1
                i -= 1
            i += 1
        if count == 0:
            count = 100
        else:
            rows += 1
        return [rows, count]

    # 819
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        words = {}
        length = len(paragraph)
        # loop through to the first letter
        i = 0
        while i < length and not paragraph[i].isalpha():
            i += 1
        left = i
        right = left + 1
        # find words
        while right < length:
            if not paragraph[right].isalpha():
                temp = paragraph[left:right].lower()
                # count
                if temp in words:
                    words[temp] = words[temp] + 1
                else:
                    words[temp] = 1
                # to the next letter
                i = right + 1
                while i < length and not paragraph[i].isalpha():
                    i += 1
                left = i
                right = left
            right += 1
        if right - 1 < length and paragraph[right - 1].isalpha():
            temp = paragraph[left:right].lower()
            # count
            if temp in words:
                words[temp] = words[temp] + 1
            else:
                words[temp] = 1
        # filter out banned
        for word in banned:
            if word in words:
                del words[word]
        # find max
        maxCount = 0
        ans = ""
        for key, val in words.items():
            if val > maxCount:
                maxCount = val
                ans = key
        return ans

    # 821
    def shortestToChar(self, s: str, c: str) -> List[int]:
        ans = []
        i = 0
        length = len(s)
        while i < length:
            if s[i] == c:
                ans.append(0)
            else:
                left = i - 1
                right = i + 1
                while left > -1 or right < length:
                    if s[max(0, left)] == c:
                        ans.append(i - left)
                        break
                    if s[min(length - 1, right)] == c:
                        ans.append(right - i)
                        break
                    left -= 1
                    right += 1
            i += 1
        return ans
    
    # 824
    def toGoatLatin(self, sentence: str) -> str:
        vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
        ans = []
        count = 1
        curWord = []
        curLen = 0
        for char in sentence:
            # proccess word
            if char == ' ':
                if not curWord[0] in vowels:
                    temp = curWord[0]
                    for i in range(curLen - 1):
                        curWord[i] = curWord[i + 1]
                    curWord[curLen - 1] = temp
                    curWord.append("ma")
                else:
                    curWord.append("ma")
                for x in range(count):
                    curWord.append('a')
                ans.append(''.join(curWord))
                count += 1
                curWord = []
                curLen = 0
                continue
            curWord.append(char)
            curLen += 1
        if not curWord[0] in vowels:
            temp = curWord[0]
            for i in range(curLen - 1):
                curWord[i] = curWord[i + 1]
            curWord[curLen - 1] = temp
            curWord.append("ma")
        else:
            curWord.append("ma")
        for x in range(count):
            curWord.append('a')
        ans.append(''.join(curWord))
        return " ".join(ans)
    
    # 830
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        left = 0
        right = left + 1
        length = len(s)
        curLen = 1
        ans = []
        while right < length:
            if s[right] != s[left]:
                if curLen >= 3:
                    ans.append([left, right - 1])
                curLen = 1
                left = right
            else:
                curLen += 1
            right += 1
        if curLen >= 3 and left != right - 1 and s[left] == s[right - 1]:
            ans.append([left, right - 1])
        return ans
    
    # 832
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        rowLen = len(image[0])
        rows = len(image)
        i = 0
        while i < rows:
            left = 0
            right = rowLen - 1
            while left <= right:
                if left == right:
                    image[i][left] = 0 if image[i][left] == 1 else 1
                else:
                    image[i][left], image[i][right] = image[i][right], image[i][left]
                    image[i][left] = 0 if image[i][left] == 1 else 1
                    image[i][right] = 0 if image[i][right] == 1 else 1
                left += 1
                right -= 1
            i += 1
        return image
    
    # 844
    def backspaceCompare(self, s: str, t: str) -> bool:
        word = []
        for char in s:
            if char == '#':
                if word:
                    word.pop()
            else:
                word.append(char)
        word1 = []
        for char in t:
            if char == '#':
                if word1:
                    word1.pop()
            else:
                word1.append(char)
        return word == word1
    
    # 846
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        length = len(hand)
        if length % groupSize != 0:
            return False
        # count the hands
        cards = {}
        for num in hand:
            cards[num] = 1 + cards.get(num, 0)
        while cards:
            cardMin = min(cards.keys())
            for i in range(groupSize):
                if not cardMin in cards:
                    return False
                cards[cardMin] = cards[cardMin] - 1
                if cards[cardMin] == 0:
                    del cards[cardMin]
                cardMin += 1
        return True
    
    # 853
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        sorted = []
        for i in range(len(position)):
            sorted.append((position[i], speed[i]))
        sorted.sort(reverse=True)
        ans = 0
        curMax = 0
        for i in range(len(position)):
            time = (target - sorted[i][0]) / sorted[i][1]
            if time > curMax:
                ans += 1
                curMax = time
        return ans
    
    # 875
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        piles.sort()
        left = 1
        right = piles[-1]
        minSpeed = right
        while left <= right:
            speed = (left + right) // 2
            time = sum([math.ceil(pile / speed) for pile in piles])
            if time > h:
                left = speed + 1
            elif time <= h:
                minSpeed = min(minSpeed, speed)
                right = speed - 1
        return minSpeed

    # 1002
    def commonChars(self, words: List[str]) -> List[str]:
        # count all occurences
        chars = []
        for word in words:
            curChars = {}
            for char in word:
                curChars[char] = 1 + curChars.get(char, 0)
            chars.append(curChars)
        # count common occurences
        res = chars[0]
        for i in range(1, len(chars)):
            comp = chars[i]
            for char, val in res.items():
                if char in comp:
                    res[char] = min(comp[char], val)
                else:
                    res[char] = 0
        ans = []
        for key, val in res.items():
            for i in range(val):
                ans.append(key)
        return ans
    
    # 1071
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        len1 = len(str1)
        len2 = len(str2)
        if len1 < len2:
            i = len1
            while i > 0:
                if str1[0:i] * (len2 // i) == str2 and str1[0:i] * (len1 // i) == str1:
                    return str1[0:i]
                i -= 1
        else:
            i = len2
            while i > 0:
                if str2[0:i] * (len1 // i) == str1 and str2[0:i] * (len2 // i) == str2:
                    return str2[0:i]
                i -= 1
        return ""
    
    # 1431
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        maxCandies = candies[0]
        for i in candies:
            if i > maxCandies:
                maxCandies = i
        res = []
        for i in candies:
            if i + extraCandies >= maxCandies:
                res.append(True)
            else:
                res.append(False)
        return res
    
    # 1679
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        right = len(nums) - 1
        count = 0
        while left < right:
            curSum = nums[left] + nums[right]
            if curSum > k:
                right -= 1
            elif curSum < k:
                left += 1
            else:
                count += 1
                left += 1
                right -= 1
        return count
    
    # 1768
    def mergeAlternately(self, word1: str, word2: str) -> str:
        len1 = len(word1)
        len2 = len(word2)
        res = []
        idx1 = 0
        idx2 = 0
        while idx1 < len1 and idx2 < len2:
            if idx2 < idx1:
                res.append(word2[idx2])
                idx2 += 1
            else:
                res.append(word1[idx1])
                idx1 += 1
        while idx1 < len1:
            res.append(word1[idx1])
            idx1 += 1
        while idx2 < len2:
            res.append(word2[idx2])
            idx2 += 1
        return ''.join(res)
    
    # 2109
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        cur = 0
        i = 0
        res = []
        while cur < len(spaces) and i < len(s):
            if i == spaces[cur]:
                res.append(' ')
                cur += 1
            else:
                res.append(s[i])
                i += 1
        return ''.join(res) + s[i:]
    
    # 2140
    def mostPoints(self, questions: List[List[int]]) -> int:
        length = len(questions)
        i = length - 1
        points = [questions[i][0]]
        i -= 1
        while i >= 0:
            prevIdx = questions[i][1] + i + 1
            if prevIdx >= length:
                points.append(max(questions[i][0], points[-1]))
            else:
                points.append(max(points[-1], questions[i][0] + points[length - i - 2 - questions[i][1]]))
            i -= 1
        return points[-1]
    
    # 2278
    def percentageLetter(self, s: str, letter: str) -> int:
        return s.count(letter) // len(s) * 100
    
    # 2575
    def divisibilityArray(self, word: str, m: int) -> List[int]:
        div = []
        cur = 0
        for i in word:
            cur = (cur * 10 + int(i)) % m
            div.append(1 if cur == 0 else 0)
        return div

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def getDepth(root, depth):
            if not root:
                return depth
            return max(getDepth(root.left, depth + 1), getDepth(root.right, depth + 1))
        return getDepth(root, 0)
    
    # 2716
    def minimizedStringLength(self, s: str) -> int:
        charSet = [0] * 26
        for char in s:
            charSet[ord(char) - ord('a')] = 1
        return sum(charSet)
    
    # 2873
    def maximumTripletValue(self, nums: List[int]) -> int:
        length = len(nums)
        maxVal = 0
        for i in range(length):
            for j in range(i+1, length):
                for k in range(j+1, length):
                    maxVal = max(maxVal, (nums[i] - nums[j]) * nums[k])
        return maxVal
    
# Test
solution = Solution()
node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = None

ln1 = ListNode(5)
ln2 = ListNode(5)

print(solution.subarraysDivByK([23,2,4,6,7], 6))

