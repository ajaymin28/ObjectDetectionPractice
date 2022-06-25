from fileinput import filename
import xml.etree.ElementTree as ET
import os
from logging import Logger

class XMLParser:

    def __init__(self, xmlfile) -> None:

        self.Log = Logger(__name__)
        self.root = None
        self.xmlfile = xmlfile



        if self.xmlfile is not None:

            if os.path.exists(xmlfile):
                self.root = ET.parse(xmlfile).getroot()
            else:
                self.Log.error(f"{xmlfile} doesnt exist")


    
    def getImageSize(self, nodeName="size"):
        w,h,c = None,None,None
        if self.xmlfile is not None:
            for tag in self.root.findall(nodeName):
                w = int(tag.find('width').text)
                h = int(tag.find('height').text)
                c = int(tag.find('depth').text)
        else:
            self.Log.error("Input file error")

        return w,h,c


    def __get_tag_value(self,tagName):
        for tag in self.root.findall("filename"):
            return tag.text

    def __parse_bb(self, node):
        xmin = int(node.find("xmin").text)
        ymin = int(node.find("ymin").text)
        xmax = int(node.find("xmax").text)
        ymax = int(node.find("ymax").text)
        return xmin,ymin,xmax,ymax


    def getAnnotations(self, annotationsTag="object"):

        annotations = {}

        if self.xmlfile is not None:
            fileName = self.__get_tag_value("filename")
            ImageFileSize = self.getImageSize()
            annotations.update({fileName: {"annotations": [], "size": ImageFileSize}})

            for tag in self.root.findall(annotationsTag):
                annotation = {}
                className = tag.find("name").text
                bbroot = tag.find("bndbox")
                xmin,ymin,xmax,ymax = self.__parse_bb(bbroot)
                annotation.update({"class": className, "box": [xmin,ymin,xmax,ymax]})
                annotations[fileName]["annotations"].append(annotation)

        return fileName, annotations


