#pragma once

#include <string>
#include <vector>
#include "Vertex.h"
#include "referential.h"

#include <itkImage.h>

using namespace std;

class Mesh
{
public:
	Mesh();
	Mesh(const Mesh& m);
	virtual ~Mesh();

	virtual void clear();

	virtual int addPoint(Vertex *v); // ajoute le point dans le vecteur et retourne sa position
	virtual int addPointLocal(Vertex *v);
	virtual void removeLastPoints(int number);
	virtual vector<Vertex*>& getListPoints() { return points_; };
	virtual vector<Vertex*>& getListLocalPoints() { return pointsLocal_; };
	virtual void addTriangle(int p1, int p2, int p3);
	virtual void removeLastTriangles(int number);
	virtual vector<int>& getListTriangles() { return triangles_; };
	virtual vector<Vertex*>* getListTrianglesBarycentre() { return &trianglesBarycentre_; };

	virtual int getNbrOfPoints() { return (int)points_.size(); };
	virtual int getNbrOfTriangles() { return (int)triangles_.size()/3; };

	virtual void save(string filename);
	virtual void saveBYU(string filename);
	virtual void read(string filename);

	virtual void setSelected(bool sel) { is_selected = sel; };
	virtual bool isSelected() const { return is_selected; };
	virtual void setDraw(bool draw) { to_draw = draw; };
	virtual bool toDraw() { return to_draw; };
	virtual void setLabel(int label) { label_ = label; };
	virtual int getLabel() { return label_; };

	virtual void setReferential(const Referential& ref, bool local=false);
	virtual void computeTrianglesBarycentre();
	virtual void transform(CMatrix4x4 transformation);
	virtual void transform(CMatrix4x4 transformation, CVector3 rotationPoint);

	virtual CMatrix4x4 ICP(Mesh* sp);

	virtual void computeConnectivity();
	virtual vector< vector<int> >& getConnectiviteTriangles() { return connectiviteTriangles_; };
	virtual vector< vector<int> >& getNeighbors() { return neighbors_; };

	virtual void decimation(float nb);
	virtual void smoothing(int numberOfIterations);
	virtual void subdivision(int numberOfSubdivision=1, bool computeFinalMesh=true);

	virtual double distanceMean(Mesh *sp);

	virtual void computeMeshNormals();
    
    virtual double computeStandardDeviationFromPixelsInside(itk::Image<double,3>::Pointer image);

protected:
	void calculateLocalPoints(CMatrix4x4 transformation);

	vector<Vertex*> points_;
	vector<Vertex*> pointsLocal_;
	vector<int> triangles_; // Triangles sous format [T1_p1 T1_p2 T1_p3 T2_p1 ...]
	vector<Vertex*> trianglesBarycentre_;
	vector< vector<int> > connectiviteTriangles_; // liste, pour chaque point, des triangles qui le contiennent
	vector< vector<int> > neighbors_; // liste, pour chaque point, de ses voisins

	int label_;

	bool repereLocalbool;
	Referential repereLocal_;

private:
	vector<Vertex*> markers_;

	bool is_selected;
	bool to_draw;
};

