
import numpy as np
import pandas as pd

class DSAListNode:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None

    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

    def getNext(self):
        return self.next

    def getPrev(self):
        return self.prev

    def setNext(self, newNext):
        self.next = newNext

    def setPrev(self, newPrev):
        self.prev = newPrev


class DSALinkedList(DSAListNode):
    def __init__(self):
        self.head = None
        self.tail = None
        

    def insertFirst(self, value):
        self.newNd = DSAListNode(value)
        if self.isEmpty():
            self.head = self.newNd
            self.tail = self.newNd
        else:
            self.newNd.setNext(self.head)
            self.head.setPrev(self.newNd)
            self.head = self.newNd
        print("Element inserted at front: ", value)
        return ""

    def insertLast(self, value):
        self.newNd = DSAListNode(value)
        if self.isEmpty():
            self.head = self.newNd
            self.tail = self.newNd
        else:
            self.newNd.setPrev(self.tail)
            self.tail.setNext(self.newNd)
            self.tail = self.newNd
        return ""

    def remove(self, value):
        current = self.head
        prev = None

        while current:
            if current.value == value:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next

                if current.next is None:
                    self.tail = prev

                return
            prev = current
            current = current.next


    def contains(self, value):
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False


    def isEmpty(self):
        return self.head == None

    def peekFirst(self):
        if self.isEmpty():
            return IndexError("List is empty!")
        else:
            nodeValue = self.head.getValue()
        print("First element: ", nodeValue)
        return ""

    def peekLast(self):
        if self.isEmpty():
            raise IndexError("List is empty!")
        else:
            peek = self.tail.getValue()
        print("Last/tail element: ", peek)
        return ""

    def removeFirst(self):
        if self.isEmpty():
            return IndexError("List is Empty!")
        else:
            nodeValue = self.head.getValue()
            self.head = self.head.getNext()
        print("Element removed: ", nodeValue)
        return ""

    def removeLast(self):
        if self.isEmpty():
            return IndexError("List is Empty!")
        elif self.head.getNext() == None:
            nodeValue = self.head.getValue()
            self.head = None
            print(nodeValue)
        else:
            prevNd = None
            currNd = self.head
            while currNd.getNext() != None:
                prevNd = currNd
                currNd = currNd.getNext()
            prevNd.setNext(None)
            self.tail = prevNd
            nodeValue = currNd.getValue()
        print("Element removed: ", nodeValue)
        return ""

    def display(self):
        currNd = self.head
        while currNd != None:
            print(currNd.getValue())
            currNd = currNd.getNext()
        return ""

    def is_empty(self):
        return self.head is None
    
    
class DSAStack(DSALinkedList):
    def __init__(self):
        self.stack = DSALinkedList()

    def push(self, value):
        self.stack.insertLast(value)

    def pop(self):
        self.stack.removeLast()

    def top(self):
        return self.stack.peekLast()

    def display(self):
        print("\n<- STACK DISPLAY ->")
        disp = self.stack.display()
        print(disp)
        return ""


class DSADeque:
    def __init__(self):
        self.items = DSALinkedList()  # Use your DSALinkedList as the underlying data structure

    def enqueue(self, item):
        self.items.insertLast(item)

    def dequeue(self):
        if not self.is_empty():
            item = self.items.peekFirst()
            self.items.removeFirst()
            return item
        return None

    def is_empty(self):
        return self.items.is_empty()
    
    


class DSAGraphNode:
    def __init__(self, label, value=None):
        self.label = label
        self.value = value
        self.adjacent_nodes = DSALinkedList()

    def add_neighbor(self, neighbor):
        self.adjacent_nodes.insertLast(neighbor)

    def get_neighbors(self):
        neighbors = DSALinkedList()  # Create a linked list to store neighbors
        current = self.adjacent_nodes.head
        while current is not None:
            neighbor = current.getValue()
            if isinstance(neighbor, str):  # Check if the neighbor is a label (string)
                neighbors.insertLast(neighbor)  # Insert the neighbor into the linked list
            current = current.getNext()
        return neighbors



class DSAGraph:
    def __init__(self):
        self.nodes = DSALinkedList()
        self.vertex_count = 0
        self.edge_count = 0

    def add_vertex(self, label, value=None):
        if not self.has_vertex(label):
            self.nodes.insertLast(DSAGraphNode(label, value))
            self.vertex_count += 1


    def remove_vertex(self, label):
        node = self.find_node(label)

        if node is not None:
            # Get the linked list of neighbors
            neighbors = node.get_neighbors()

            # Remove the vertex from the list of nodes
            self.nodes.remove(node)

            # Remove all edges associated with the vertex
            current = neighbors.head
            while current is not None:
                neighbor_label = current.getValue()
                self.remove_edge(label, neighbor_label)

                # Remove the reverse edge for the neighbor
                neighbor_node = self.find_node(neighbor_label)
                neighbor_node.adjacent_nodes.remove(label)

                current = current.getNext()

            self.vertex_count -= 1
            print(f"Shop with number {label} has been removed.")
        else:
            print(f"Shop with number {label} does not exist.")





    def add_edge(self, label1, label2):
        node1 = self.find_node(label1)
        node2 = self.find_node(label2)

        if node1 is not None and node2 is not None:
            if not node1.adjacent_nodes.contains(label2):
                node1.add_neighbor(label2)
                self.edge_count += 1
                print(f"Edge between '{label1}' and '{label2}' added.")

                # Add the reverse edge for an undirected graph
                if not node2.adjacent_nodes.contains(label1):
                    node2.add_neighbor(label1)
                    print(f"Edge between '{label2}' and '{label1}' added.")

    def has_vertex(self, label):
        return self.find_node(label) is not None

    def find_node(self, label):
        curr = self.nodes.head
        while curr is not None:
            node = curr.getValue()
            if node.label == label:
                return node
            curr = curr.getNext()
        return None

    def get_vertex_count(self):
        return self.vertex_count

    def get_edge_count(self):
        return self.edge_count
    

    def remove_edge(self, label1, label2):
        node1 = self.find_node(label1)
        node2 = self.find_node(label2)

        if node1 is not None and node2 is not None:
            neighbors1 = node1.get_neighbors()
            neighbors2 = node2.get_neighbors()

            if label2 in neighbors1:
                neighbors1.remove(label2)
                self.edge_count -= 1

            if label1 in neighbors2:
                neighbors2.remove(label1)


    def display_as_list(self):
        curr = self.nodes.head
        while curr is not None:
            node = curr.getValue()
            neighbors = node.get_neighbors()
            neighbors_str = ""

            current = neighbors.head
            while current is not None:
                neighbor = current.getValue()
                neighbors_str += neighbor
                current = current.getNext()
                if current is not None:
                    neighbors_str += ", "
            
            print(f"{node.label}: {neighbors_str}")
            curr = curr.getNext()



    def display_as_matrix(self):
        # Find the maximum label to determine the size of the matrix
        max_label = -1
        curr = self.nodes.head
        while curr is not None:
            node = curr.getValue()
            max_label = max(max_label, int(node.label))
            curr = curr.getNext()

        # Ensure that max_label is at least 1
        max_label = max(max_label, 1)

        # Initialize the matrix with zeros
        matrix = np.zeros((max_label, max_label), dtype=int)

        # Fill in the matrix
        curr = self.nodes.head
        while curr is not None:
            node = curr.getValue()
            neighbors = node.get_neighbors()
            node_label = int(node.label)

            current = neighbors.head
            while current is not None:
                neighbor = current.getValue()
                neighbor_label = int(neighbor)

                # Set matrix elements to 1 for existing edges
                matrix[node_label - 1][neighbor_label - 1] = 1  # Adjust indices to start from 0

                current = current.getNext()

            curr = curr.getNext()

        # Display the matrix
        for row in matrix:
            row_str = ' '.join([str(elem) for elem in row])
            print(row_str)



class DEQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            return None

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)
    

class DSAHashEntry:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class DSAHashTable:
    def __init__(self, size=100, upper_threshold=0.7, lower_threshold=0.3):
        self.size = size
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.num_elements = 0
        self.table = [None] * size


    def get(self, key):
        entry_index = self.find(key)
        if entry_index is not None:
            return self.table[entry_index].value
        else:
            raise KeyError("Key not found")

    def fnv_hash(self, key):
        # FNV-1a hash function
        FNV_offset_basis = 14695981039346656037
        FNV_prime = 1099511628211
        hash_val = FNV_offset_basis
        for char in key:
            hash_val ^= ord(char)
            hash_val *= FNV_prime
        return hash_val % self.size

    def elf_hash(self, key):
        # ELF hash function
        hash_val = 0
        for char in key:
            hash_val = (hash_val << 4) + ord(char)
            x = hash_val & 0xF0000000
            if x != 0:
                hash_val ^= (x >> 24)
            hash_val &= ~x
        return hash_val % self.size

    def double_hashing(self, key, attempt):
        hash1 = self.fnv_hash(key)
        hash2 = self.elf_hash(key)
        return (hash1 + attempt * hash2) % self.size

    def resize(self, new_size):
        old_table = self.table
        self.size = new_size
        self.table = [None] * new_size
        self.num_elements = 0

        for entry in old_table:
            if entry is not None:
                self.put(entry.key, entry.value)

    def find(self, key):
        attempt = 0
        index = self.double_hashing(key, attempt)
        while self.table[index] is not None:
            if self.table[index].key == key:
                return index
            attempt += 1
            index = self.double_hashing(key, attempt)
        return None

    def needs_size_reduction(self):
        return (self.num_elements / self.size) < self.lower_threshold

    def put(self, key, value):
        if (self.num_elements + 1) / self.size > self.upper_threshold:
            # If the load factor exceeds the upper threshold, resize the table
            new_size = self.size * 2
            self.resize(new_size)
             # If the load factor exceeds the lower threshold, resize the table
        if self.needs_size_reduction():
            new_size = max(self.size // 2, 1)  
            self.resize(new_size)

    

        # Check for duplicate key before entering the loop
        for entry in self.table:
            if entry and entry.key == key:
                raise KeyError("Duplicate key found")

        attempt = 0
        index = self.double_hashing(key, attempt)

        # Loop until an empty slot or a slot with the same key is found
        while self.table[index] is not None and self.table[index].key != key:
            attempt += 1
            index = self.double_hashing(key, attempt)

            # If all slots are checked and the table is full, resize it
            if attempt >= self.size:
                new_size = self.size * 2
                self.resize(new_size)
                # Re-calculate index after resizing
                index = self.double_hashing(key, attempt)
                attempt = 0  # Reset attempt counter

        # Insert the entry into the table
        self.table[index] = DSAHashEntry(key, value)
        self.num_elements += 1


    def has_key(self, key):
        return self.find(key) is not None

    def remove(self, key):
        entry_index = self.find(key)
        if entry_index is not None:
            self.table[entry_index] = None
            self.num_elements -= 1


class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, value):
        self.heap.append(value)
        self._heapify_up()

    def pop(self):
        if len(self.heap) == 0:
            return None

        if len(self.heap) == 1:  # Corrected 'this' to 'self'
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down()

        return root

    def _heapify_up(self):
        index = len(self.heap) - 1
        while index > 0:
            parent_index = (index - 1) // 2
            if self.heap[parent_index][0] <= self.heap[index][0]:  # Change to '<' for ascending order
                break
            self.heap[parent_index], self.heap[index] = self.heap[index], self.heap[parent_index]
            index = parent_index

    def _heapify_down(self):
        index = 0
        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest = index

            if (
                left_child_index < len(self.heap)
                and self.heap[left_child_index][0] < self.heap[smallest][0]  # Change to '<' for ascending order
            ):
                smallest = left_child_index

            if (
                right_child_index < len(self.heap)
                and self.heap[right_child_index][0] < self.heap[smallest][0]  # Change to '<' for ascending order
            ):
                smallest = right_child_index

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest




class ShopNavigationSystem:
    def __init__(self):
        self.graph = DSAGraph()  # Initialize a graph for shop representation
        self.shop_categories = {}  # Initialize an empty dictionary for shop categories
        self.category_hash_table = DSAHashTable() # Define shop_categories as a class attribute




    def add_shop(self, shop_number, shop_name, category, location, rating):

        category = category.lower()
        if category not in self.shop_categories:
            self.shop_categories[category] = []
        self.shop_categories[category].append((shop_number, shop_name, location, rating))

        if not self.graph.has_vertex(shop_number):
            while True:
                try:
                    rating = int(rating)
                    if 1 <= rating <= 5:
                        shop_info = {
                            "Shop Name": shop_name,
                            "Category": category,
                            "Location": location,
                            "Rating": rating
                        }
                        self.graph.add_vertex(shop_number, shop_info)
                        print(f"Shop '{shop_name}' (Number: {shop_number}) added.")
                        break  # Exit the loop when a valid rating is provided
                    else:
                        print("Rating must be between 1 and 5. Please try again.")
                        rating = input("Enter a valid rating: ")
                except ValueError:
                    print("Rating must be a valid integer. Please try again.")
                    rating = input("Enter a valid rating: ")
        else:
            print(f"Shop '{shop_name}' (Number: {shop_number}) already exists.")
            

    

    def remove_shop(self, shop_number):
        try:
            if self.graph.has_vertex(shop_number):
                # Get the shop category from the graph
                shop_info = self.graph.find_node(shop_number).value
                category = shop_info.get("Category")

                # Remove the shop from the graph using the shop_number as the label
                self.graph.remove_vertex(shop_number)

                # Update the shop categories dictionary
                if category in self.shop_categories and len(self.shop_categories[category]) > 0:
                    self.shop_categories[category] = [shop for shop in self.shop_categories[category] if shop[0] != shop_number]

                print(f"Shop with number {shop_number} removed.")
            else:
                print(f"Shop with number {shop_number} does not exist.")
        except Exception as e:
            print(f"An error occurred while removing the shop: {e}")

    
    def update_shop_info(self, shop_number, shop_name, category, location, rating):
        try:
            if self.graph.has_vertex(shop_number):
                while True:
                    try:
                        rating = int(rating)
                        if 1 <= rating <= 5:
                            shop_info = {
                                "Shop Name": shop_name,
                                "Category": category,
                                "Location": location,
                                "Rating": rating
                            }
                            self.graph.find_node(shop_number).value = shop_info
                            print(f"Shop information updated for '{shop_name}' (Number: {shop_number}.")

                            # Update the shop_categories dictionary as well
                            for cat, shop_list in self.shop_categories.items():
                                for index, shop in enumerate(shop_list):
                                    if shop[0] == shop_number:
                                        shop_list[index] = (shop_number, shop_name, location, rating)
                            break  # Exit the loop when the update is successful
                        else:
                            print("Rating must be between 1 and 5. Please try again.")
                            rating = input("Enter a valid rating: ")
                    except ValueError:
                        print("Rating must be a valid integer. Please try again.")
                        rating = input("Enter a valid rating: ")
            else:
                print(f"Shop with number {shop_number} does not exist.")
        except Exception as e:
            print(f"An error occurred while updating shop information: {e}")





    def add_connection(self, source_shop, destination_shop):
        if self.graph.has_vertex(source_shop) and self.graph.has_vertex(destination_shop):
            self.graph.add_edge(source_shop, destination_shop)
            print(f"Connection added between Shop {source_shop} and Shop {destination_shop}.")
        else:
            print("One or both of the shops do not exist.")



    def remove_connection(self, source_shop, destination_shop):
        if self.graph.has_vertex(source_shop) and self.graph.has_vertex(destination_shop):
            source_node = self.graph.find_node(source_shop)
            current = source_node.adjacent_nodes.head
            found = False
            
            while current is not None:
                neighbor = current.getValue()
                if neighbor == destination_shop:
                    source_node.adjacent_nodes.remove(destination_shop)
                    found = True
                    break
                current = current.getNext()
            
            if found:
                destination_node = self.graph.find_node(destination_shop)
                current = destination_node.adjacent_nodes.head
                while current is not None:
                    neighbor = current.getValue()
                    if neighbor == source_shop:
                        destination_node.adjacent_nodes.remove(source_shop)
                        break
                    current = current.getNext()
                
                print(f"Connection removed between Shop {source_shop} and Shop {destination_shop}.")
            else:
                print(f"No connection found between Shop {source_shop} and Shop {destination_shop}.")
        else:
            print("One or both of the shops do not exist.")




    def find_path_dfs(self, source_shop, destination_shop):
        try:
            source_shop = str(source_shop)
            destination_shop = str(destination_shop)

            if not self.graph.has_vertex(source_shop) or not self.graph.has_vertex(destination_shop):
                print("Invalid source or destination shop. Please check the shop numbers.")
                return None

            def dfs(node, path, visited):
                if node == destination_shop:
                    return path + [node]

                visited.add(node)
                neighbors = self.graph.find_node(node).get_neighbors()
                current = neighbors.head
                while current is not None:
                    neighbor = current.getValue()
                    if neighbor not in visited:
                        new_path = dfs(neighbor, path + [node], visited)
                        if new_path:
                            return new_path
                    current = current.getNext()

            visited = set()
            path = dfs(source_shop, [], visited)
            if path:
                return path
            else:
                print("No valid path found from the source to the destination.")
                return None
        except Exception as e:
            print(f"An error occurred while finding the path: {e}")
        


    def find_path_bfs(self, source_shop, destination_shop):
        try:
            source_shop = str(source_shop)
            destination_shop = str(destination_shop)

            if not self.graph.has_vertex(source_shop) or not self.graph.has_vertex(destination_shop):
                print("Invalid source or destination shop. Please check the shop numbers.")
                return None

            queue = DEQueue()
            queue.enqueue([source_shop])

            visited = set()

            while not queue.is_empty():
                path = queue.dequeue()
                node = path[-1]

                if node == destination_shop:
                    return path

                if node not in visited:
                    visited.add(node)
                    neighbors = self.graph.find_node(node).get_neighbors()
                    current = neighbors.head
                    while current is not None:
                        neighbor = current.getValue()
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.enqueue(new_path)
                        current = current.getNext()

            print("No valid path found from the source to the destination.")
            return None
        except Exception as e:
            print(f"An error occurred while finding the path: {e}")



    def compare_paths(self, source_shop, destination_shop):
        dfs_path = self.find_path_dfs(source_shop, destination_shop)
        bfs_path = self.find_path_bfs(source_shop, destination_shop)

        if dfs_path is not None and bfs_path is not None:
            dfs_distance = len(dfs_path) - 1  # The distance is the number of edges (steps) between nodes
            bfs_distance = len(bfs_path) - 1

            print("DFS Path:", "->".join(map(str, dfs_path)), "Distance:", dfs_distance)
            print("BFS Path:", "->".join(map(str, bfs_path)), "Distance:", bfs_distance)

            if dfs_distance == bfs_distance:
                print("Both DFS and BFS paths have the same distance.")
            elif dfs_distance < bfs_distance:
                print("DFS provides the shorter path.")
            else:
                print("BFS provides the shorter path.")



    def search_by_category(self, category):
        try:
            # Convert the category to lowercase (case-insensitive)
            category = category.lower()

            if category in self.shop_categories:
                category_shops = self.shop_categories[category]
                print(f"Shops in the category '{category}':")
                for shop_number, shop_name, location, rating in category_shops:
                    print(f"Shop {shop_number}: {shop_name} at {location} (Rating: {rating})")
            else:
                print(f"No matching shops found in the category '{category}'.")
        except Exception as e:
            print(f"An error occurred while searching for shops by category: {e}")




    def search_shops_using_heaps(self, category):
        category = category.lower()
        if category not in self.shop_categories:
            print(f"No shops found in the '{category}' category.")
            return

        shops_in_category = self.shop_categories[category]

        if not shops_in_category:
            print(f"No shops found in the '{category}' category.")
            return

        # Use a max-heap to sort shops by rating
        sorted_shops = MaxHeap()
        for shop in shops_in_category:
            shop_number, shop_name, location, rating = shop
            try:
                rating = int(rating)  # Ensure rating is treated as an integer
                sorted_shops.push((-rating, shop_name, shop_number, location))  # Negate rating for max-heap
            except ValueError:
                print(f"Invalid rating found for shop '{shop_name}' (Number: {shop_number}). Skipping.")

        print(f"Shops in the '{category}' category sorted by rating:")

        while True:
            shop = sorted_shops.pop()
            if shop is None:
                break
            rating, shop_name, shop_number, location = shop
            print(f"Shop Name: {shop_name} (Number: {shop_number}), Location: {location}, Rating: {-rating}")  # Negate rating back to positive









    def read_data_file(self, file_path):
        try:
            data = pd.read_excel(file_path, sheet_name=0)  # Read data from the first sheet (shops)
            connections_data = pd.read_excel(file_path, sheet_name=1)  # Read data from the second sheet (connections)

            for index, row in data.iterrows():
                shop_number = str(row['shop_number'])
                shop_name = str(row['shop_name'])
                category = str(row['shop_category'])
                location = str(row['shop_location'])
                rating = str(row['shop_rating'])

                self.add_shop(shop_number, shop_name, category, location, rating)

            for index, row in connections_data.iterrows():
                source_shop = str(row['source_shop'])
                destination_shop = str(row['destination_shop'])

                if self.graph.has_vertex(source_shop) and self.graph.has_vertex(destination_shop):
                    self.add_connection(source_shop, destination_shop)
            
            print("Shops and connections added from the Excel file.")
        except Exception as e:
            print(f"An error occurred while reading the Excel file: {e}")


            



    def display_menu(self):
        print("\nInteractive Shop Navigation Menu:")
        print("(a) Add Shop")
        print("(b) Remove Shop")
        print("(c) Update Shop Information")
        print("(d) Add Edge (Connection)")
        print("(e) Remove Edge (Connection)")
        print("(f) Display")
        print("(g) Path Analysis")
        print("(h) Read Test Data (Excel)")
        print("(i) Search Shops by (Category) Hashing")
        print("(J) Search Shops by (Category) using Heaps")
        print("(q) Quit")



    def handle_display_option(self):
        while True:
            print("\nDisplay Options:")
            print("(1) Display as List")
            print("(2) Display as Matrix")
            print("(b) Back to Main Menu")

            choice = input("Enter your choice: ").strip().lower()

            if choice == '1':
                self.graph.display_as_list()
            elif choice == '2':
                self.graph.display_as_matrix()
            elif choice == 'b':
                break
            else:
                print("Invalid choice. Please select a valid option.")


    def handle_find_paths_option(self):
        while True:
            print("\nFind Paths Options:")
            print("(1) Breadth-First Search (BFS)")
            print("(2) Depth-First Search (DFS)")
            print("(3) Shortest Path (Comparison)")
            print("(b) Back to Main Menu")

            choice = input("Enter your choice: ").strip().lower()

            if choice == '1':
                # Call the BFS path-finding method
                source_shop = input("Enter source shop number: ").strip()
                destination_shop = input("Enter destination shop number: ").strip()
                self.find_and_display_path_bfs(source_shop, destination_shop)

            elif choice == '2':
                # Call the DFS path-finding method
                source_shop = input("Enter source shop number: ").strip()
                destination_shop = input("Enter destination shop number: ").strip()
                self.find_and_display_path_dfs(source_shop, destination_shop)

            elif choice == '3':
                # Call the path comparison method
                source_shop = input("Enter source shop number: ").strip()
                destination_shop = input("Enter destination shop number: ").strip()
                self.compare_paths(source_shop, destination_shop)

            elif choice == 'b':
                break
            else:
                print("Invalid choice. Please select a valid option.")

    def find_and_display_path_bfs(self, source_shop, destination_shop):
        bfs_path = self.find_path_bfs(source_shop, destination_shop)
        if bfs_path is not None:
            print("BFS Path:", "->".join(bfs_path))

    def find_and_display_path_dfs(self, source_shop, destination_shop):
        dfs_path = self.find_path_dfs(source_shop, destination_shop)
        if dfs_path is not None:
            print("DFS Path:", "->".join(dfs_path))


    def run(self):
        while True:
            self.display_menu()
            choice = input("Enter your choice: ").strip().lower()

            if choice == 'a':
                shop_number = input("Enter shop number: ").strip()
                shop_name = input("Enter shop name: ").strip()
                category = input("Enter shop category: ").strip()
                location = input("Enter shop location: ").strip()
                rating = input("Enter shop rating (1-5 stars): ").strip()
                self.add_shop(shop_number, shop_name, category, location, rating)

            elif choice == 'b':
                shop_number = input("Enter shop number to remove: ").strip()
                self.remove_shop(shop_number)

            elif choice == 'c':
                shop_number = input("Enter shop number to update: ").strip()
                shop_name = input("Enter updated shop name: ").strip()
                category = input("Enter updated shop category: ").strip()
                location = input("Enter updated shop location: ").strip()
                rating = input("Enter updated shop rating (1-5 stars): ").strip()
                self.update_shop_info(shop_number, shop_name, category, location, rating)

            elif choice == 'd':
                source_shop = input("Enter source shop number: ").strip()
                destination_shop = input("Enter destination shop number: ").strip()
                self.add_connection(source_shop, destination_shop)

            elif choice == 'e':
                source_shop = input("Enter source shop number: ").strip()
                destination_shop = input("Enter destination shop number: ").strip()
                self.remove_connection(source_shop, destination_shop)


            elif choice == "f":
                self.handle_display_option()


            elif choice == 'g':
                self.handle_find_paths_option()

            elif choice == 'h':
                file_path = input("Enter the path to the Excel data file: ").strip()
                self.read_data_file(file_path)


            elif choice == 'i':
                search_category = input("Enter the category to search: ").strip()
                self.search_by_category(search_category)

            elif choice == 'j':
                search_category = input("Enter the category to search: ").strip()
                self.search_shops_using_heaps(search_category)

            elif choice == 'q':
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please select a valid option.")


if __name__ == "__main__":
    # Create an instance of ShopNavigationSystem
    shop_system = ShopNavigationSystem()

    # Start the interactive menu
    shop_system.run()
