"use server";

import prisma from "@/lib/prisma";
import { PathData } from "@/types/path";

export async function addPathData(pathData: PathData) {
  return await prisma.data.create({
    data: pathData,
  });
}
