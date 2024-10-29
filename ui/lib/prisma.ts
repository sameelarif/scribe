import { PrismaClient } from "@prisma/client";

export default new PrismaClient({
    log: ['info', 'error', 'query', 'warn']
})