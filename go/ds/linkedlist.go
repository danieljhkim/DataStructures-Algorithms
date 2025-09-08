package ds

type Node struct {
	Value int
	Next  *Node
}

type SinglyLinkedList struct {
	Head *Node
	Size int
}

func NewSinglyLinkedList() *SinglyLinkedList {
	return &SinglyLinkedList{
		Head: nil,
		Size: 0,
	}
}

func (l *SinglyLinkedList) InsertFront(value int) {
	newNode := &Node{
		Value: value,
		Next:  l.Head,
	}
	l.Head = newNode
	l.Size++
}

func (l *SinglyLinkedList) InsertBack(value int) {
	newNode := &Node{Value: value}
	
	if l.Head == nil {
		l.Head = newNode
		l.Size++
		return
	}
	
	current := l.Head
	for current.Next != nil {
		current = current.Next
	}
	
	current.Next = newNode
	l.Size++
}

func (l *SinglyLinkedList) Delete(value int) bool {
	if l.Head == nil {
		return false
	}
	
	if l.Head.Value == value {
		l.Head = l.Head.Next
		l.Size--
		return true
	}
	
	current := l.Head
	for current.Next != nil && current.Next.Value != value {
		current = current.Next
	}
	
	if current.Next == nil {
		return false
	}
	
	current.Next = current.Next.Next
	l.Size--
	return true
}