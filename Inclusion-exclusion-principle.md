There is an interesting thing i had never noticed about a problem when using Inclusion Exclusion principle.
This problem can be breifly described as:
  In a party, guests are required to put their hats in one place together. how many different combos that each one pick a hat that doesn't belong to him when they leave the party?
  
Previously, i only consider about solving this by dynamic programming, here, we denote S(n) as #combos when there are n people in the party.
Thus, S(n) = (n-1)S(n-1) + (n-1)S(n-2). Here, we can easily use this formula to compute results from a single PC.

There is another solution for this problem from a very different viewpoint, someone try to introduce "Inclusion-Exclusion-Principle" here.
#combo = #total - #at_least_one_same (#total means all types and is equal to n!, #at_least_one_same means the number of combo that there is at least one person who get his own hat).
Typically, It is indicated that #at_least_one_same = #at_least_1_same \and #at_least_2 \and #at_least_3_same \and ... \and #at_least_n_same. (#at_S()least_n_same means the number of combos that the n-th person get his own hat correctly).  
Thus, we can easily derive:
  
  S(n) = n! - 1Cn(n-1)! + 2Cn(n-2)! - 3Cn(n-3)! +- ... +- nCn(0!)
 
so, according to the formular above, S(1) = 0, S(2) = 1, S(3) = 2, S(4) = 9, S(5) = 44, S(6) = 265 ...

we make some modification and give an approximation for the formular above:

  S(n) = n!(1-1+1/2!-1/3!+1/4!+...+1/n!)

According to Taylor expansion, we know e^-1 = 1-1+1/2!+...+1/n!.
Thus,

  S(n) = n!/e
  
This is the approximation form of our solution.
