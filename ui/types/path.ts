export interface Point {
    x: number
    y: number
    timestamp: number
}
  
export interface Box {
    id: string
    x: number
    y: number
}

export interface PathData {
    start: Point
    end: Point
    path: Point[]
}