import numpy as np  
import os 

def nfunc(str):
    if str.endswith('\n'):
        return str[:-1]
    else:
        return str
def jiou(num):
    if num%2==0:
        return 1
    else:
        return -1

def calculate_angle_N(v):
    v1 = v[0]
    v2 = v[1]
    a = (v1 * v2).sum(axis=-1)
    b = (v1 ** 2).sum(axis=-1)
    c = (v2 ** 2).sum(axis=-1)
    d = (b * c) ** 0.5
    cos = a / d
    cos = np.around(cos, decimals=12)
    theta = np.arccos(cos) * 180 / np.pi
    return theta


def convert_to_spherical(cartesian):
    r = np.linalg.norm(cartesian, axis=-1)
    theta = np.arccos(np.take(cartesian, 2, axis=-1) / r) * 180 / np.pi
    phi = np.arctan2(np.take(cartesian, 1, axis=-1), np.take(cartesian, 0, axis=-1)) * 180 / np.pi
    spherical = np.stack([r, theta, phi], axis=-1)
    return spherical


def Dist(vector):
    return (vector ** 2).sum(axis=-1) ** 0.5


class Input(object):
    def __init__(self, lattice_vector=np.array([]), Pb=np.array([]), Br=np.array([]), Br_id=np.array([]),
                 lattice_label=np.array([]), Br_N=np.array([]), Pb_N=np.array([]), Pb_lattice_label=np.array([]),
                 Pb_id=np.array([]), twoD_id=np.array([]), Pb_name='Pb', Br_name='Br'):
        self.lattice_vector = lattice_vector  
        self.Pb = Pb  # Coord of Pb atoms    shape:(nPb,3)
        self.Br = Br  # Coord of Br atoms    shape:(nBr,3)
        self.Br_id = Br_id  # Index of Br atoms    shape:(nPb,6)
        self.Br_lattice_label = lattice_label  # Record which lattice Br should be translated to    shape:(nBr,nPb)
        self.Br_N = Br_N  # Coord of all 27 lattices    shape:(nBr,27,3)
        self.Pb_N = Pb_N  # Coord of extended Pb atoms    shape:(nPb,27,3)
        self.Pb_lattice_label = Pb_lattice_label  # Record which lattice Pb should be translated to    shape:(nPb,nBr)
        self.Pb_id = Pb_id  # Similar to Br_id, record the id of Pb atoms that belong to each Br, since each Br is assumed to belong to two Pb atoms, the shape is (nBr,2)
        self.twoD_id = twoD_id  # Classify Br atoms in the plane and those not in the plane, [0] contains Br atoms in the plane, [1] contains Br atoms not in the plane    shape:(2,nBr/2)
        self.vector_Pb_Br = np.array([])  # Vector from each Pb to Br (sorted by方位)    shape:(nPb,4,3)
        self.vector_Br_Pb = np.array([])  # Not sorted    shape:(4,2,3)
        self.Br_id_2d = np.array([])  # Record the id of Br atoms connected to each Pb (only record Br atoms in the plane, the same number does not represent the same Br as in total Br_id)(sorted by方位)    shape:(2,4)
        self.Pb_id_2d = np.array([])  # Record the id of Pb atoms connected to each Br (only record Br atoms in the plane, the same number does not represent the same Br as in total Br_id)(not sorted by方位)    shape:(2,4,2)
        self.Pb_vectors_id = np.array([])  # Record the id of Pb atoms connected to each Br (only record Br atoms in the plane, the same number does not represent the same Br as in total Br_id)(sorted by方位)    shape:(2,4)

        self.Pb_name = Pb_name  # Name of Pb atoms    shape:(nPb,)
        self.Br_name = Br_name  # Name of Br atoms    shape:(nBr,)
        self.file_name = 'geometry.in'  # Name of the file to be read
        self.file_path = ''  # Path

        self.vector_vertical_Pb_Br = np.array([])  # Only for 3D, vertical Pb-Br vector (sorted by方位)

        self.Others_name_list = np.array([])  # Store the name information of other atoms
        self.Others_list = np.array([])  # Store the coordinate information of other atoms

    def read_geometry(self, file_name='geometry.in', file_path='', Pb_name='Pb', Br_name='Br'):  # Record the lattice vector, Pb, Br original coordinates
        with open(os.path.join(file_path, file_name), "rt") as f:
            file_lines = list(f.readlines())
        lattice_vector_list = []
        Pb_list = []
        Br_list = []
        Others_list = []
        Others_name_list = []

        if 'I' in file_path:
            Br_name = 'I'
        elif 'Cl' in file_path:
            Br_name = 'Cl'

        for line in file_lines:
            if line.startswith('lattice_vector'):
                line_list = nfunc(line).split()
                line_list.pop(0)
                line_list = list(map(float, line_list))
                lattice_vector_list.append(line_list)
            elif line.startswith('atom'):
                if line.endswith(Br_name + '\n'):
                    line_list = line.split()
                    line_list.pop(0)
                    line_list.pop()
                    line_list = list(map(float, line_list))
                    Br_list.append(line_list)
                elif line.endswith(Pb_name + '\n'):
                    line_list = line.split()
                    line_list.pop(0)
                    line_list.pop()
                    line_list = list(map(float, line_list))
                    Pb_list.append(line_list)
                else:
                    line_list = line.split()
                    line_list.pop(0)
                    Others_name_list.append(line_list.pop())
                    line_list = list(map(float, line_list))
                    Others_list.append(line_list)
        lattice_vector_list = np.array(lattice_vector_list)
        self.Br = np.array(Br_list)
        self.Pb = np.array(Pb_list)
        self.lattice_vector = lattice_vector_list
        self.file_name = file_name
        self.file_path = file_path
        self.Pb_name = Pb_name
        self.Br_name = Br_name
        self.Others_name_list = np.array(Others_name_list)
        self.Others_list = np.array(Others_list)

    def calculate_lattice(self):
        # Turn the initial cell into the surrounding lattices, a total of 27 cells
        lattice_vector = self.lattice_vector
        Pb = self.Pb
        Br = self.Br
        # Traverse 27 vectors and add them to the original coordinates of the initial lattice to obtain the coordinates of 27 lattices
        lst = np.array(
            [i * lattice_vector[0] + j * lattice_vector[1] + k * lattice_vector[2] for i in [-1, 0, 1]
             for j in [-1, 0, 1] for k in [-1, 0, 1]])  # The combination of 27 lattice vectors
        # Calculate the distance (square) between each extended Br and each Pb in the initial lattice, and find the shortest distance and the corresponding lattice index for each Br in all lattices
        Br_New = Br[:, np.newaxis, :]
        Br1 = Br_New + lst
        Br1_New = Br1[:, :, np.newaxis, :]
        dist2 = (Br1_New - Pb) ** 2
        dist2 = dist2.sum(axis=3)
        label1 = np.argmin(dist2, axis=1)  # When axis=1, it is the traversal of each cell
        dist2 = dist2.min(axis=1)  # Calculate the corresponding distance
        # Find the index of the 6 closest Br to each Pb (the index in all Br, sorted by the initial order of Br in the file) as the Br bonded to it
        Br_id = np.array([np.sort(np.argsort(dist2[:, i])[:6]) for i in range(len(Pb))])    
    
        # Do the same operation on Pb, find the shortest distance and the corresponding lattice index between each Pb and each Br in all lattices
        Pb_New = Pb[:, np.newaxis, :]
        PbN = Pb_New + lst
        Pb1_New = PbN[:, :, np.newaxis, :]
        Pb_dist2 = (Pb1_New - Br) ** 2
        Pb_dist2 = Pb_dist2.sum(axis=3)
        label2 = np.argmin(Pb_dist2, axis=1)
        Pb_dist2 = Pb_dist2.min(axis=1)
        #Find out the index of the 2 closest Pb to each Br (the index in all Pb, sorted by the initial order of Pb in the file) as the Pb bonded to it
        Pb_id = np.array([np.sort(np.argsort(Pb_dist2[:, i])[:2]) for i in range(len(Br))])
        self.Br_id = Br_id
        self.Br_lattice_label = label1
        self.Br_N = Br1
        self.Pb_N = PbN
        self.Pb_lattice_label = label2
        self.Pb_id = Pb_id

    def calculate_2D(self):  # Classify Br located on the plane and outside the plane, only one key with Pb on the outside, and two keys with Pb on the plane
        # The following work is mainly to count the number of times each Br appears in Br_id, Br appears once in the outside plane, and appears twice in the plane
        twoD_Br_id_list = [[], []] #twoD_Br_id_list[1] is the index of Br outside the plane, twoD_Br_id_list[0] is the index of Br on the plane
        # Flatten Br_id
        twoDD = np.reshape(self.Br_id, (len(self.Br_id[0]) * len(self.Br_id)))
        counts = np.bincount(twoDD)
        for i in range(len(self.Br)):
            if counts[i] == 1:
                twoD_Br_id_list[1].append(i)
            elif counts[i] == 2:
                twoD_Br_id_list[0].append(i)
        self.twoD_id = np.array(twoD_Br_id_list)

    def reorder_lattice_vector(self):  # Reorder the lattice vector, put the lattice vector between the planes at the last position (this will not change the value of the three lattice vectors themselves and the original coordinate order)
        # Calculate a Pb-Br bond outside the plane, and find the angle between it and the three lattice vectors, the vector corresponding to the angle less than 45 degrees or greater than 135 degrees is the lattice vector perpendicular to the plane
        lattice_vector_list = list(self.lattice_vector)
        nBr1 = self.twoD_id[1][0]
        # If nBr1 is in Br_id[i], it means that the i-th Pb is bonded to the nBr1-th Br
        for i in range(len(self.Pb)):
            if nBr1 in self.Br_id[i]:
                nPb1 = i
        # Calculate the vector perpendicular to the plane, then calculate the angle between all lattice vectors and the vector perpendicular to the plane
        xBr1 = self.Br_N[nBr1, self.Br_lattice_label[nBr1, nPb1]]
        vec1 = xBr1 - self.Pb[nPb1]
        anglist = calculate_angle_N([vec1, self.lattice_vector])
        # Record the number of times to rotate
        nCvector = 0
        for i in range(3):
            if anglist[i] < 45 or anglist[i] > 135:
                nCvector = i
        # Rotate the lattice vector perpendicular to the plane to the last position
        if nCvector == 0:
            lattice_vector_list.append(lattice_vector_list.pop(0))
        elif nCvector == 1:
            lattice_vector_list.append(lattice_vector_list.pop(0))
            lattice_vector_list.append(lattice_vector_list.pop(0))
        self.lattice_vector = np.array(lattice_vector_list)

    def calculate_vectors(self):  # Calculate the vector of each atom's Pb-Br bond
        # The following line of code is a bit long, x_list_Br first traverses Pb (j), then traverses the Br on the plane belonging to this Pb (i), and gets the coordinates of Br
        # Create a numpy array x_list_Br, used to store the coordinates of the Br atoms bonded to the Pb atoms in the initial cell on the plane
        x_list_Br = np.array(
            [[self.Br_N[i, self.Br_lattice_label[i, j]] for i in np.intersect1d(self.twoD_id[0], self.Br_id[j])] for j
             in range(len(self.Pb))])
        # Create a numpy array vector_Pb_Br_list, used to store the coordinate difference between the Pb atom and the Br atom
        vector_Pb_Br_list = np.array([x_list_Br[i] - self.Pb[i] for i in range(len(x_list_Br))])
        # Create a numpy array x_list_Pb, used to store the coordinates of the Pb atoms bonded to the Br atoms in the initial cell
        # j traverses all the Br on the plane, i traverses the Pb belonging to this Br, and gets the coordinates of Pb
        x_list_Pb = np.array(
            [[self.Pb_N[i, self.Pb_lattice_label[i, j]] for i in self.Pb_id[j]] for j in self.twoD_id[0]])
        vector_Br_Pb_list = np.array([x_list_Pb[i] - self.Br[self.twoD_id[0][i]] for i in range(len(x_list_Pb))])

        # When 3D, in order to find the adjacent atoms of each atom on the plane, the label needs to be reset
        Pb_id_2d = self.Pb_id[self.twoD_id] #Separate the Br on the plane and the Br outside the plane (the id was recorded even though it was not judged whether the Br was on the plane, so the distance to the Br outside the plane was also recorded)#Pb_id_2d[0] records the Br on the plane
        Br_id = self.Br_id
        Br_id_2d = []
        # Convert the 3D index to the 2D index, if the 2D index is i, then the 3D index is twoD_id[0][i]
        for i in Br_id:
            new_row = []
            for j in i:
                if np.isin(j, self.twoD_id[0]):
                    index = np.where(self.twoD_id[0] == j)[0][0]
                    new_row.append(index)
            Br_id_2d.append(np.array(new_row))
        Br_id_2d = np.array(Br_id_2d)

        # The vector along the initial direction and the vector in the z direction
        lv = self.lattice_vector
        vertical_vector = np.cross(lv[0], lv[1])
        # The initial direction is the sum of the two plane lattice vectors
        vector_0 = lv[0] + lv[1]
        # vector1 is the vector perpendicular to the initial direction and the z direction
        if len(vector_0) == 3:
            vector_1 = np.cross(vertical_vector, vector_0)  
        else:
            # Rotate 90 degrees counterclockwise
            vector_1 = np.array([-vector_0[1], vector_0[0]])

        # lst records the index of the Br atoms bonded to the Pb atoms on each plane (four Br indices)
        lst = []
        # Find the corresponding id
        for i in range(len(self.Pb)):
            id_list1 = np.argsort(calculate_angle_N([vector_0, vector_Pb_Br_list[i]]))
            id_list2 = np.argsort(calculate_angle_N([vector_1, vector_Pb_Br_list[i]]))
            # lst each row's four elements are: the index of the Br atom with the smallest angle to vector_0, the index of the Br atom with the smallest angle to vector_1, the index of the Br atom with the largest angle to vector_0, the index of the Br atom with the largest angle to vector_1
            lst.append(np.array([id_list1[0], id_list2[0], id_list1[-1], id_list2[-1]]))
        lst = np.array(lst)

        # Reorder the vectors according to the方位
        total_list = [vector_Pb_Br_list[i][lst[i]] for i in range(len(self.Pb))]
        total_list = np.array(total_list)

        # Although the following two lines of code are redundant in most cases, when there are more than 4 Br atoms on one layer, Br_id_2d is not a simple [0 1 2 3] array, so the index of lst is not the same as the index of total_list
        total_list_id = [Br_id_2d[i][lst[i]] for i in range(len(self.Pb))]
        total_list_id = np.array(total_list_id)

        # Calculate which Pb each Pb on the plane is connected to
        Pb_vectors_lst = []
        for i in range(len(total_list_id)):
            vectlst = []
            for j in range(4):
                # Find the same Br_id in the total_list_id of all bonded Br atoms in the opposite direction, then the index of Pb_new is the new Pb that the original Pb is connected to after skipping one Br
                Pb_n_New = np.argwhere(total_list_id[:, (j + 2) % 4] == total_list_id[i, j])[0][0]
                vectlst.append(Pb_n_New)
            Pb_vectors_lst.append(np.array(vectlst))
        Pb_vectors_lst = np.array(Pb_vectors_lst)
        self.Pb_vectors_id = Pb_vectors_lst
        self.vector_Pb_Br = total_list
        self.vector_Br_Pb = vector_Br_Pb_list
        self.Br_id_2d = total_list_id
        self.Pb_id_2d = Pb_id_2d

    # calculate all vertical Pb_Br bonds, and sort them in space
    # Here, the index of the Br outside the plane is not recorded, only the vector of the Pb-Br bond outside the plane is calculated
    def calculate_vertical_Pb_Br_vectors(self):
        # Record the coordinates of the Br outside the plane
        x_list_Br = np.array(
            [[self.Br_N[i, self.Br_lattice_label[i, j]] for i in np.intersect1d(self.twoD_id[1], self.Br_id[j])] for j
             in range(len(self.Pb))])
        vector_Pb_Br_list = np.array([x_list_Br[i] - self.Pb[i] for i in range(len(x_list_Br))])
        lv = self.lattice_vector
        vertical_vector = np.cross(lv[0], lv[1])

        # Record the index of the Br outside the plane corresponding to which vector
        lst = []
        for i in range(len(self.Pb)):
            id_list1 = np.argsort(calculate_angle_N([vertical_vector, vector_Pb_Br_list[i]]))
            lst.append(np.array([id_list1[0], id_list1[-1]]))
        lst = np.array(lst)
        total_list = [vector_Pb_Br_list[i][lst[i]] for i in range(len(self.Pb))]
        total_list = np.array(total_list)
        self.vector_vertical_Pb_Br = total_list

    # Create geometry
    def creat_geomotry(self,theta=np.array([20.75,14.0]),l=8.3563709259,Br_name='Br',Pb_name='Pb',l_theta=0,l_a=1.0):
        theta=np.array(theta)
        theta=theta*np.pi/180
        lattice_vector = np.array([[l, 0, 0], [0, l, 0], [0, 0, 80]])
        bond_lenth=(1+np.tan(np.mean(theta))**2/2)**0.5*l*2**(-1.5)
        # One Pb is placed in the middle, and one Pb is placed at the origin
        Pb = np.array([[0, 0, 0], 0.5 * lattice_vector[0] + 0.5 * lattice_vector[1]])  +0.5*lattice_vector[2]
        # In order to simplify the code, I used various complex calculation methods, but the disadvantage is that it will forget the code after a long time
        # The Br on the plane, j=0 and j=1 represent the Br below and above, i=0 and i=0.5 represent the Br on the left and right
        # The term "np.array([0,0,-jiou(j)*l*0.25*np.tan(theta[-j])])" in the brackets represents the upward and downward movement of the Br, the left side is lowered and the right side is raised
        Br_parallel = np.array(
            [(0.25 + i) * lattice_vector[0] + (0.25 + j*0.5) * lattice_vector[1]+np.array([0,0,-jiou(j)*l*0.25*np.tan(theta[-j])]) for i in [0, 0.5] for j in [0, 1]])  +0.5*lattice_vector[2]
        Br_vertical = np.array([Pb[j] + i * np.array([0, bond_lenth*np.sin(np.mean(theta))*jiou(j), bond_lenth*np.cos(np.mean(theta))]) for i in [1, -1] for j in [0,1]])
        Br = np.concatenate((Br_parallel, Br_vertical))
        Cs = np.array([0.5 * lattice_vector[j] + i * np.array([0, 0, bond_lenth]) for i in [-1, 1] for j in [0, 1]])  +0.5*lattice_vector[2]
        l_theta=l_theta*np.pi/180
        shift_matrix=np.array([[np.sqrt(l_a),0,0],[np.sin(l_theta)/np.sqrt(l_a),np.cos(l_theta)/np.sqrt(l_a),0],[0,0,1]])
        self.lattice_vector = np.dot(lattice_vector,shift_matrix)
        self.Pb = np.dot(Pb,shift_matrix)
        self.Br = np.dot(Br,shift_matrix)
        self.Br_name = Br_name
        self.Pb_name = Pb_name
        self.Others_list = np.dot(Cs,shift_matrix)
        self.Others_name_list = ["Cs"] * 8

    def rereorder_lattice_vector(self,num=2):  # reorder the lattice vector
        lattice_vector_list = list(self.lattice_vector)
        for i in range(num):
            lattice_vector_list.append(lattice_vector_list.pop(0))
        self.lattice_vector = np.array(lattice_vector_list)

    def add_Cs(self):
        cs_o=np.mean(self.Pb,axis=0)
        # For different handed coordinate systems, the sign of the two lattice vectors needs to be changed (replace the + with - in the next line)
        vect=(self.lattice_vector[0]+self.lattice_vector[1])/4
        Cs1=np.mean(self.vector_vertical_Pb_Br,axis=0)+cs_o+vect
        Cs2=np.mean(self.vector_vertical_Pb_Br,axis=0)+cs_o-vect
        self.Others_list=np.concatenate((Cs1,Cs2))
        self.Others_name_list=np.array(['Cs']*len(np.concatenate((Cs1,Cs2))))

    def convert_lattice_vector(self):
        lv=self.lattice_vector
        self.lattice_vector=np.array([lv[1],-lv[0],lv[2]])

    def calculate_Cs(self):  # This function assumes that the position of Cs in the file is correct on both sides of the inorganic layer
        # Return the distance of each Cs to the inorganic layer of Pb
        Cs_list = []
        for i in range(len(self.Others_name_list)):
            if self.Others_name_list[i] == 'Cs':
                Cs_list.append(self.Others_list[i])
        # The array of points that need to be calculated to the plane distance
        Cs_list = np.array(Cs_list)
        # The point on the plane
        Pb_vector = self.Pb.mean(axis=0)
        # Lattice vector (the first two vectors are on the plane, but the third vector is not perpendicular to the first vector, the vector is not normalized)
        lv = self.lattice_vector

        # Calculate the normal vector of the plane
        normal_vector = np.cross(lv[0], lv[1])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 单位化

        # Calculate the distance of each Cs point to the Pb plane
        vectors_to_Pb = Cs_list - Pb_vector  # The vector of each Cs to Pb_vector
        distances = np.abs(np.dot(vectors_to_Pb, normal_vector))  # Calculate the distance to the plane

        return distances

    def move_Cs(self, lenth=0.1):  # This function assumes that other atoms only have Cs
        # The Cs on both sides of the plane move towards the plane or away from the plane by lenth length
        Cs_list = []
        for i in range(len(self.Others_name_list)):
            if self.Others_name_list[i] == 'Cs':
                Cs_list.append(self.Others_list[i])

        # The array of points that need to be calculated to the plane distance
        Cs_list = np.array(Cs_list)

        # The point on the plane
        Pb_vector = self.Pb.mean(axis=0)

        # Lattice vector (the first two vectors are on the plane, but the third vector is not perpendicular to the first vector, the vector is not normalized)
        lv = self.lattice_vector

        # Calculate the normal vector of the plane (the cross product of lv[0] and lv[1])
        normal_vector = np.cross(lv[0], lv[1])
        normal_vector /= np.linalg.norm(normal_vector)  # Normalize the normal vector

        # Calculate the distance of each Cs point to the plane
        for i in range(len(Cs_list)):
            # Calculate the distance of Cs point to the plane
            vector_to_plane = Cs_list[i] - Pb_vector
            distance_to_plane = np.dot(vector_to_plane, normal_vector)

            # Determine which side of the plane the Cs atom is on, and move lenth length
            if distance_to_plane > 0:
                # Cs atom on one side of the plane, move towards the plane
                Cs_list[i] -= lenth * normal_vector
            else:
                # Cs atom on the other side of the plane, move away from the plane
                Cs_list[i] += lenth * normal_vector

        # Update the position of Cs atom in Others_list
        self.Others_list = np.array([atom if name != 'Cs' else Cs_list[i]
                                     for i, (name, atom) in enumerate(zip(self.Others_name_list, self.Others_list))])

# projection processing
class Two_D_Input(object):
    def __init__(self, lattice_vector=np.array([]), Pb=np.array([]), Br=np.array([]), Br_id=np.array([]),
                 lattice_label=np.array([]), Br_N=np.array([]), Pb_N=np.array([]), Pb_lattice_label=np.array([]),
                 Pb_id=np.array([]), twoD_Br_id=np.array([])):
        self.lattice_vector = lattice_vector  # Lattice vector (three-dimensional)
        self.Pb = Pb  # The coordinates of Pb
        self.Br = Br  # The coordinates of Br
        self.Br_id = Br_id  # The id of Br belonging to the first Pb and the second Pb
        self.Br_lattice_label = lattice_label  # Record which lattice should be translated to
        self.Br_N = Br_N  # The coordinates of all 27 Br atoms
        self.Pb_N = Pb_N  # The coordinates of the extended Pb
        self.Pb_lattice_label = Pb_lattice_label  # Record which lattice should be translated to
        self.Pb_id = Pb_id  # Similar to Br_id
        self.xOy = np.array([])  # The vectors of x and y axes in three-dimensional coordinates
        self.twoD_id = np.array([np.arange(4)])  # Classify the Br on the plane and the Br not on the plane, [0] is the Br on the plane, [1] is the Br not on the plane
        self.vector_Pb_Br = np.array([])  # The vector of each Pb to Br (sorted by方位)
        self.vector_Br_Pb = np.array([])  # Not sorted
        self.Br_id_2d = np.array([])  # Record the id of Br connected to each Pb (only record the Br on the plane, the same number with the total Br_id does not represent the same Br)(sorted by方位)
        self.Pb_id_2d = np.array([])  # Record the id of Pb connected to each Br (only record the Br on the plane, the same number with the total Br_id does not represent the same Br)(not sorted by方位)
        self.Pb_vectors_id = np.array([])  # Record the id of Pb connected to each Pb after skipping 1 Br (sorted by方位)

        self.Pb_name = 'Pb'  # The name of the atom corresponding to the Pb position (default is Pb)
        self.Br_name = 'Br'  # The name of the atom corresponding to the Br position (default is Br)
        self.file_name = 'geometry.in'  # The name of the file to be read
        self.file_path = ''  # Path

    def read_geometry(self, three_D_input=Input()):
        # Because the plane formed by the average of the two Pb is parallel to the plane formed by the lattice vectors ab, the coordinates of the atoms projected onto the two planes only differ by a constant, so the plane formed by ab can be used as the projection plane, and the calculated bond length and bond angle remain unchanged
        # The parameters used: lattice_vector, Br, Pb, twoD_id[0]

        vertical_vector = np.cross(three_D_input.lattice_vector[0], three_D_input.lattice_vector[1])

        # Translate the entire coordinate system to the average of the c components of the two Pb
        c_vector = three_D_input.lattice_vector[-1]
        Br = three_D_input.Br
        Pb = three_D_input.Pb
        # Calculate the new x-axis vector and y-axis vector
        x_vector = three_D_input.lattice_vector[0]
        y_vector = np.cross(x_vector, vertical_vector)

        # In order to obtain better x-axis and y-axis, the number with the largest absolute value must be positive
        if x_vector[np.argmax(np.abs(x_vector))] < 0:
            x_vector = -x_vector
        if y_vector[np.argmax(np.abs(y_vector))] < 0:
            y_vector = -y_vector

        # Ensure the order of x-axis and y-axis remains unchanged
        if np.argmax(x_vector) > np.argmax(y_vector):
            x_vector, y_vector = y_vector, x_vector

        # Make the length of the two vectors equal to 1
        x_vector = x_vector / (x_vector ** 2).sum(axis=-1) ** 0.5
        y_vector = y_vector / (y_vector ** 2).sum(axis=-1) ** 0.5
        vertical_vector = vertical_vector / (vertical_vector ** 2).sum(axis=-1) ** 0.5

        # First establish a new three-dimensional coordinate system, then calculate the new coordinates, and then directly delete the coordinates in the vertical direction (Ai may have bugs)
        xyz_arr = np.array([x_vector, y_vector, vertical_vector])
        # Transpose
        xyz_arr_inv = np.transpose(xyz_arr)
        Br_projection = np.dot(Br[three_D_input.twoD_id[0]], xyz_arr_inv)
        Br_projection = np.delete(Br_projection, -1, axis=-1)
        Pb_projection = np.dot(Pb, xyz_arr_inv)
        Pb_projection = np.delete(Pb_projection, -1, axis=-1)
        lattice_vector_New = np.dot(three_D_input.lattice_vector[:2], xyz_arr_inv)
        lattice_vector_New = np.delete(lattice_vector_New, -1, axis=-1)

        self.lattice_vector = lattice_vector_New
        self.xOy = np.array([x_vector, y_vector])
        self.Br = Br_projection
        self.Pb = Pb_projection
        self.vertical_vector = vertical_vector
        self.file_name = three_D_input.file_name
        self.file_path = three_D_input.file_path
        self.Pb_name = three_D_input.Pb_name
        self.Br_name = three_D_input.Br_name
        self.twoD_id = np.array([np.arange(len(Pb) * 2)])

    def calculate_lattice(self):
        # Calculate the 9 surrounding lattices
        lst = np.array(
            [i * self.lattice_vector[0] + j * self.lattice_vector[1] for i in [-1, 0, 1]
             for j in [-1, 0, 1]])  # The combination of 9 lattice vectors
        Br = self.Br
        Pb = self.Pb
        Br_New = Br[:, np.newaxis, :]
        Br1 = Br_New + lst
        Br1_New = Br1[:, :, np.newaxis, :]
        dist2 = (Br1_New - Pb) ** 2
        dist2 = dist2.sum(axis=3)
        label1 = np.argmin(dist2, axis=1)
        dist2 = dist2.min(axis=1)
        Br_id = np.array([np.sort(np.argsort(dist2[:, i])[:4]) for i in range(len(Pb))])

        Pb_New = Pb[:, np.newaxis, :]
        Pb1 = Pb_New + lst
        Pb1_New = Pb1[:, :, np.newaxis, :]
        Pb_dist2 = (Pb1_New - Br) ** 2
        Pb_dist2 = Pb_dist2.sum(axis=3)
        label2 = np.argmin(Pb_dist2, axis=1)
        Pb_dist2 = Pb_dist2.min(axis=1)
        Pb_id = np.array([np.sort(np.argsort(Pb_dist2[:, i])[:2]) for i in range(len(Br))])
        self.Br_id = Br_id
        self.Br_lattice_label = label1
        self.Br_N = Br1
        self.Pb_N = Pb1
        self.Pb_lattice_label = label2
        self.Pb_id = Pb_id

    def calculate_vectors(self):
        x_list_Br = np.array(
            [[self.Br_N[i, self.Br_lattice_label[i, j]] for i in np.intersect1d(self.twoD_id[0], self.Br_id[j])] for j
             in range(len(self.Pb))])
        vector_Pb_Br_list = np.array([x_list_Br[i] - self.Pb[i] for i in range(len(x_list_Br))])
        x_list_Pb = np.array(
            [[self.Pb_N[i, self.Pb_lattice_label[i, j]] for i in self.Pb_id[j]] for j in self.twoD_id[0]])
        vector_Br_Pb_list = np.array([x_list_Pb[i] - self.Br[self.twoD_id[0, i]] for i in range(len(x_list_Pb))])

        vector_0 = self.lattice_vector[0] + self.lattice_vector[1]
        if len(vector_0) == 3:
            vector_1 = np.cross(self.lattice_vector[-1], vector_0)
        else:
            # Rotate 90 degrees counterclockwise
            vector_1 = np.array([-vector_0[1], vector_0[0]])

        Br_id_2d = self.Br_id
        lst = []
        for i in range(len(self.Pb)):
            id_list1 = np.argsort(calculate_angle_N([vector_0, vector_Pb_Br_list])[i])
            id_list2 = np.argsort(calculate_angle_N([vector_1, vector_Pb_Br_list])[i])
            lst.append(np.array([id_list1[0], id_list2[0], id_list1[-1], id_list2[-1]]))
        lst = np.array(lst)
        total_list = [vector_Pb_Br_list[i][lst[i]] for i in range(len(self.Pb))]
        total_list = np.array(total_list)

        total_list_id = [Br_id_2d[i][lst[i]] for i in range(len(self.Pb))]
        total_list_id = np.array(total_list_id)

        Pb_vectors_lst = []
        for i in range(len(total_list_id)):
            vectlst = []
            for j in range(4):
                Pb_n_New = np.argwhere(total_list_id[:, (j + 2) % 4] == total_list_id[i, j])[0][0]
                vectlst.append(Pb_n_New)
            Pb_vectors_lst.append(np.array(vectlst))
        Pb_vectors_lst = np.array(Pb_vectors_lst)
        self.Pb_vectors_id = Pb_vectors_lst
        self.vector_Pb_Br = total_list
        self.vector_Br_Pb = vector_Br_Pb_list
        self.Br_id_2d = total_list_id
        self.Pb_id_2d = self.Pb_id


# Because there are two or more Pb in some files, after examination, these Pb may belong to the same layer, Input class can only process one layer, so we need to divide them
# Return a list, each element is an Input type, representing each layer
def devide_layers_Input(file_name='geometry.in', file_path='', Pb_name='Pb', Br_name='Br'):
    with open(os.path.join(file_path, file_name), "rt") as f:
        file_lines = list(f.readlines())
    lattice_vector_list = []
    Pb_list = []
    Br_list = []
    Others_list = []
    Others_name_list = []

    for line in file_lines:
        if line.startswith('lattice_vector'):
            line_list = nfunc(line).split()
            line_list.pop(0)
            line_list = list(map(float, line_list))
            lattice_vector_list.append(line_list)
        elif line.startswith('atom'):
            if line.endswith(Br_name + '\n'):
                line_list = line.split()
                line_list.pop(0)
                line_list.pop()
                line_list = list(map(float, line_list))
                Br_list.append(line_list)
            elif line.endswith(Pb_name + '\n'):
                line_list = line.split()
                line_list.pop(0)
                line_list.pop()
                line_list = list(map(float, line_list))
                Pb_list.append(line_list)
            else:
                line_list = line.split()
                line_list.pop(0)
                Others_name_list.append(line_list.pop())
                line_list = list(map(float, line_list))
                Others_list.append(line_list)
    lattice_vector_list = np.array(lattice_vector_list)

    lattice_vector = lattice_vector_list
    Pb = np.array(Pb_list)
    Br = np.array(Br_list)

    lst = np.array(
        [i * lattice_vector[0] + j * lattice_vector[1] + k * lattice_vector[2] for i in [-1, 0, 1]
         for j in [-1, 0, 1] for k in [-1, 0, 1]])
    Br_New = Br[:, np.newaxis, :]
    Br_N = Br_New + lst
    Br1_New = Br_N[:, :, np.newaxis, :]
    dist2 = (Br1_New - Pb) ** 2
    dist2 = dist2.sum(axis=-1)
    Br_lattice_label = np.argmin(dist2, axis=1)
    dist2 = dist2.min(axis=1)
    Br_id = np.array([np.sort(np.argsort(dist2[:, i])[:6]) for i in range(len(Pb))])

    Pb_New = Pb[:, np.newaxis, :]
    Pb_N = Pb_New + lst
    Pb1_New = Pb_N[:, :, np.newaxis, :]
    Pb_dist2 = (Pb1_New - Br) ** 2
    Pb_dist2 = Pb_dist2.sum(axis=-1)
    Pb_lattice_label = np.argmin(Pb_dist2, axis=1)
    Pb_dist2 = Pb_dist2.min(axis=1)
    Pb_id = np.array([np.sort(np.argsort(Pb_dist2[:, i])[:2]) for i in range(len(Br))])

    species_id = {}
    count = 0
    total_list = []
    for item in Pb_id:
        try:
            species_id[str(item)].append(count)
        except KeyError:
            species_id[str(item)] = []
            species_id[str(item)].append(count)
        finally:
            count += 1

    # Merge the extra layers
    while True:
        lst = list(species_id.keys())
        lst1 = [np.fromstring(i[1:-1], dtype=int, sep=' ') for i in lst]
        len_old = len(lst)
        for i in range(len_old):
            exit_flag = False
            for j in range(i + 1, len_old):
                if len(np.intersect1d(lst1[i], lst1[j])) != 0:
                    new_name = str(np.union1d(lst1[i], lst1[j]))
                    new_list = list(set(species_id[lst[i]]) | set(species_id[lst[j]]))

                    species_id[new_name] = new_list

                    species_id.pop(lst[i])
                    species_id.pop(lst[j])

                    exit_flag = True
                    break
            if exit_flag:
                break
        if len(species_id) == len_old:
            break
    # Process each layer
    for i in species_id:
        Pb_layerlist = np.fromstring(i[1:-1], dtype=int, sep=' ')
        Br_layerlist = species_id[i]

        a = Input()
        a.lattice_vector = lattice_vector  # Lattice vector
        a.Pb = Pb[Pb_layerlist]  # Pb coordinates
        a.Br = Br[Br_layerlist]  # Br coordinates
        a.Pb_name = Pb_name  # The name of the atom corresponding to the Pb position (default is Pb)
        a.Br_name = Br_name  # The name of the atom corresponding to the Br position (default is Br)
        a.file_name = file_name  # The name of the file to be read
        a.file_path = file_path  # Path

        # Each layer in the list stores the information of other atoms
        a.Others_name_list = np.array(Others_name_list)
        a.Others_list = np.array(Others_list)

        a.calculate_lattice()
        a.calculate_2D()
        a.reorder_lattice_vector()
        a.calculate_vectors()
        a.calculate_vertical_Pb_Br_vectors()
        total_list.append(a)
    return total_list



# Calculate the bond length
def calculate_dist(yourInput=Input()):
    return Dist(yourInput.vector_Pb_Br)


# Calculate the Pb-Br-Pb angle
def calculate_beta_angle(yourInput=Input()):
    v = yourInput.vector_Pb_Br
    if len(v) == 2:
        total_list = [calculate_angle_N([v[0, i], v[1, (i + 2) % 4]]) for i in range(4)]
        return np.array(total_list)
    else:  # There are two or more Pb on the same plane
        id = yourInput.Br_id_2d
        # Record the original Pb id and Br id
        Pb_n = 0
        Br_n = id[0, 0]
        # Record the id of the center Pb (default is the first Pb, but if there are multiple Pb, it is difficult to handle)
        Pb_centre_id = [0]
        # Record the id of the Pb that has been passed
        set0 = set()
        set0.add(0)
        # For the case where there are multiple Pb in one layer, more traversal is needed
        while True:
            for i in range(len(id[Pb_n])):
                # First find the Pb connected to the first atom
                Br_n = id[Pb_n, i]
                for j in range(len(id)):
                    if np.isin(Br_n, id[j]):
                        set0.add(j)
            for i in range(len(id)):
                # If not fully traversed, set another Pb as the center atom
                if not i in set0:
                    Pb_n = i
                    set0.add(i)
                    Pb_centre_id.append(i)
            if len(set0) == len(id):
                break
        total_list = []
        for i in range(len(Pb_centre_id)):
            Pb_n = Pb_centre_id[i]
            for j in range(4):
                # Get the id of the Pb opposite the center atom
                Pb_n_New = np.argwhere(id[:, (j + 2) % 4] == id[Pb_n, j])[0][0]
                total_list.append(calculate_angle_N([v[Pb_n, j], v[Pb_n_New, (j + 2) % 4]]))
                Pb_n = Pb_n_New
        return np.array(total_list)


# Calculate the plane Br-Pb-Br angle
def calculate_bond_angle_Pb(yourInput=Input()):
    v = yourInput.vector_Pb_Br
    total_list = []
    for i in range(len(v)):
        total_list.append([calculate_angle_N([v[i, j], v[i, (j + 1) % 4]]) for j in range(4)])
    return np.array(total_list)


# Calculate the vertical Br_Pb_Br angle
def calculate_vertical_bond_angle_Pb(yourInput=Input()):
    v = yourInput.vector_Pb_Br
    vertical_v = yourInput.vector_vertical_Pb_Br
    total_list = []
    for i in range(len(v)):
        total_list1 = []
        for k in range(len(vertical_v[i])):
            total_list1.append([calculate_angle_N([vertical_v[i, k], v[i, j]]) for j in range(4)])
        total_list.append(np.array(total_list1))
    return np.array(total_list)


# Calculate the polar coordinates of the Pb bond type=0 is to convert the original coordinates, type=1 is to use one lattice vector on the plane as the x-axis, type=2 is to use the sum of two lattice vectors on the plane as the x-axis
def convert_coordinate(vectors=[], lattice_vectors=[], type=0):
    vect = vectors
    if type:  # Only when type is not 0, the coordinate needs to be converted
        lv = lattice_vectors
        z_vector = np.cross(lv[0], lv[1])
        if type == 1:
            x_vector = lv[0]
        elif type == 2:
            x_vector = lv[0] + lv[1]
        y_vector = np.cross(z_vector, x_vector)

        # Normalize
        x_vector = x_vector / (x_vector ** 2).sum(axis=-1) ** 0.5
        y_vector = y_vector / (y_vector ** 2).sum(axis=-1) ** 0.5
        z_vector = z_vector / (z_vector ** 2).sum(axis=-1) ** 0.5

        xyz_arr = np.array([x_vector, y_vector, z_vector])
        # Transpose
        xyz_arr_inv = np.transpose(xyz_arr)
        # Convert coordinates
        vect = np.dot(vect, xyz_arr_inv)
    return convert_to_spherical(vect)

def geometry_to_X(file='geometry.in',path='',angle='rad',dimension=3, Pb_name='Pb', Br_name='Br',yourinput=None):
    if yourinput==None:
        c = devide_layers_Input(file_name=file, file_path=path, Pb_name=Pb_name, Br_name=Br_name)
        a=c[0]
    else:
        a=yourinput
    a.calculate_lattice()
    a.calculate_2D()
    a.reorder_lattice_vector()
    a.calculate_vectors()
    a.calculate_vertical_Pb_Br_vectors()
    if dimension == 3:
        Br_angle_list = calculate_beta_angle(a)
        Pb_angle_list = calculate_bond_angle_Pb(a)
        lenth_list = calculate_dist(a)
        #Cs_distances_list = a.calculate_Cs()
    elif dimension == 2:
        b = Two_D_Input()
        b.read_geometry(a)
        b.calculate_lattice()
        b.calculate_vectors()
        Br_angle_list = calculate_beta_angle(b)
        Pb_angle_list = calculate_bond_angle_Pb(b)
        lenth_list = calculate_dist(b)
    if angle == 'rad':
        Br_angle_list = Br_angle_list * np.pi / 180
        Pb_angle_list = Pb_angle_list * np.pi / 180
    Pb_angle_list=Pb_angle_list.ravel()
    lenth_list=lenth_list.ravel()
    result = np.concatenate([Br_angle_list, Pb_angle_list, lenth_list])
    return result
