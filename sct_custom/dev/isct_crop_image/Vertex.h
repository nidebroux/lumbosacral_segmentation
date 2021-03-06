#ifndef VERTEX_H
#define VERTEX_H

#include "Vector3.h"
#include "Matrix4x4.h"
#include <string>
using namespace std;

class Vertex
{
public:
	Vertex();
	Vertex(double x, double y, double z);
	Vertex(double x, double y, double z, string labels);
	Vertex(string labels);
	Vertex(CVector3 point);
	Vertex(CVector3 point, bool deform);
	Vertex(CVector3 point, CVector3 normal);
	Vertex(CVector3 point, CVector3 normal, int label);
	Vertex(const Vertex &v);
	Vertex(const Vertex &v, string labels);
	~Vertex(void);
	CVector3 getPosition();
	void setPosition(CVector3 pos);
	void setLabel(int label);
	void setLabelS(string labels);
	int getLabel();
	string getLabelString();
	void setNormal(double x, double y, double z);
	CVector3 getNormal();
	double distance(double x, double y, double z);
	double distance(Vertex& v);
	CVector3 transform(CMatrix4x4& transformation);

	Vertex& operator=(const Vertex &v);
	bool operator==(const Vertex &v);
	friend ostream& operator<<( ostream &flux, const Vertex &v );

	void setDeform(bool deform) { deform_ = deform; };
	bool hasToBeDeform() { return deform_; };
private:
	CVector3 point_;
	int label_;
	string label_s_;
	CVector3 normal_;

	bool deform_;
};

#endif // VERTEX_H
