import collections

Node = collections.namedtuple('Node', ['left', 'right', 'value'])
def contains(root, value):
    if value==root.value:
        return True
    
    if value>root.value:
        if Node.right!=None:
            return contains(Node.right,value)
        else:
            return False
    if value<root.value:
        if Node.left!=None:
            return contains(Node.left,value)
        else:
            return False
    return False
n1 = Node(value=1, left=None, right=None)
n3 = Node(value=3, left=None, right=None)
n2 = Node(value=2, left=n1, right=n3)

print(contains(n2, 2)
